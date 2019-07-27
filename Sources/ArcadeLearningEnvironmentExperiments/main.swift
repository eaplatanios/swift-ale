// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import ArcadeLearningEnvironment
import Logging
import ReinforcementLearning
import TensorFlow

struct ArcadeActorCritic: Network {
  @noDerivative public var state: None = None()

  public var conv1: Conv2D<Float> = Conv2D<Float>(
    filterShape: (8, 8, 4, 32),
    strides: (4, 4),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var conv2: Conv2D<Float> = Conv2D<Float>(
    filterShape: (4, 4, 32, 64),
    strides: (2, 2),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var conv3: Conv2D<Float> = Conv2D<Float>(
    filterShape: (3, 3, 64, 64),
    strides: (1, 1),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var denseHidden: Dense<Float> = Dense<Float>(
    inputSize: 3136,
    outputSize: 512,
    weightInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var denseAction: Dense<Float> = Dense<Float>(
    inputSize: 512,
    outputSize: 4,
    weightInitializer: orthogonal(gain: Tensor<Float>(0.01))) // TODO: Easy way to get the number of actions.
  public var denseValue: Dense<Float> = Dense<Float>(
    inputSize: 512,
    outputSize: 1,
    weightInitializer: orthogonal(gain: Tensor<Float>(1.0)))

  public init() {}

  public init(copying other: ArcadeActorCritic) {
    conv1 = other.conv1
    conv2 = other.conv2
    denseHidden = other.denseHidden
    denseAction = other.denseAction
    denseValue = other.denseValue
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<UInt8>) -> ActorCriticOutput<Categorical<Int32>> {
    let input = Tensor<Float>(input) / 255.0
    let outerDimCount = input.rank - 3
    let outerDims = [Int](input.shape.dimensions[0..<outerDimCount])
    let flattenedBatchInput = input.flattenedBatch(outerDimCount: outerDimCount)
    let conv1 = relu(self.conv1(flattenedBatchInput))
    let conv2 = relu(self.conv2(conv1))
    let conv3 = relu(self.conv3(conv2)).reshaped(to: [-1, 3136])
    let hidden = relu(denseHidden(conv3))
    let actionLogits = denseAction(hidden)
    let flattenedValue = denseValue(hidden)
    // print(flattenedValue)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}

let logger = Logger(label: "Breakout PPO")

let batchSize = 6
let emulators = (0..<batchSize).map { _ in ArcadeEmulator(game: .breakout) }
let arcadeEnvironment = ArcadeEnvironment(
  using: emulators,
  observationsType: .screen(height: 84, width: 84, format: .grayscale),
  useMinimalActionSet: true,
  episodicLives: true,
  noOpReset: .stochastic(minCount: 0, maxCount: 30),
  frameSkip: .constant(4),
  frameStackCount: 4,
  parallelizedBatchProcessing: false)
let averageEpisodeReward = AverageEpisodeReward(for: arcadeEnvironment, bufferSize: 100)
let environment = EnvironmentCallbackWrapper(
  arcadeEnvironment,
  callbacks: averageEpisodeReward.updater())

print("Number of valid actions: \(arcadeEnvironment.actionCount)")

let network = ArcadeActorCritic()
var agent = PPOAgent(
  for: environment,
  network: network,
  optimizer: AMSGrad(for: network),
  learningRateSchedule: LinearLearningRateSchedule(initialValue: 2.5e-4, slope: -2.5e-4 / 5208.0),
  maxGradientNorm: 0.5,
  advantageFunction: GeneralizedAdvantageEstimation(discountFactor: 0.99, discountWeight: 0.95),
  normalizeAdvantages: true,
  useTDLambdaReturn: true,
  clip: PPOClip(epsilon: 0.1),
  penalty: PPOPenalty(klCutoffFactor: 0.5),
  valueEstimationLoss: PPOValueEstimationLoss(weight: 0.5, clipThreshold: 0.1),
  entropyRegularization: PPOEntropyRegularization(weight: 0.01),
  iterationCountPerUpdate: 1)
for step in 0..<1000000 {
  let loss = agent.update(
    using: environment,
    maxSteps: 128 * batchSize,
    maxEpisodes: 10000 * batchSize,
    stepCallbacks: [{ (environment, trajectory) in
      // if step > 0 { try! environment.render() }
    }])
  if step % 1 == 0 {
    logger.info("Step \(step) | Loss: \(loss) | Average Episode Reward: \(averageEpisodeReward.value())")
  }
}
