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

// _RuntimeConfig.useLazyTensor = true

struct ArcadeActorCritic: Module {
  public var actionConv1: Conv2D<Float> = Conv2D<Float>(
    filterShape: (8, 8, 4, 32),
    strides: (4, 4),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var actionConv2: Conv2D<Float> = Conv2D<Float>(
    filterShape: (4, 4, 32, 64),
    strides: (2, 2),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var actionConv3: Conv2D<Float> = Conv2D<Float>(
    filterShape: (3, 3, 64, 64),
    strides: (1, 1),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var actionDenseHidden: Dense<Float> = Dense<Float>(
    inputSize: 3136,
    outputSize: 512,
    weightInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var actionDenseOutput: Dense<Float> = Dense<Float>(
    inputSize: 512,
    outputSize: 4,
    weightInitializer: orthogonal(gain: Tensor<Float>(0.01))) // TODO: Easy way to get the number of actions.

  public var valueConv1: Conv2D<Float> = Conv2D<Float>(
    filterShape: (8, 8, 4, 32),
    strides: (4, 4),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var valueConv2: Conv2D<Float> = Conv2D<Float>(
    filterShape: (4, 4, 32, 64),
    strides: (2, 2),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var valueConv3: Conv2D<Float> = Conv2D<Float>(
    filterShape: (3, 3, 64, 64),
    strides: (1, 1),
    filterInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var valueDenseHidden: Dense<Float> = Dense<Float>(
    inputSize: 3136,
    outputSize: 512,
    weightInitializer: orthogonal(gain: Tensor<Float>(sqrt(2.0))))
  public var valueDenseOutput: Dense<Float> = Dense<Float>(
    inputSize: 512,
    outputSize: 1,
    weightInitializer: orthogonal(gain: Tensor<Float>(1.0)))

  @differentiable
  public func callAsFunction(_ input: Tensor<UInt8>) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.rank - 3
    let outerDims = [Int](input.shape.dimensions[0..<outerDimCount])
    let input = Tensor<Float>(input.flattenedBatch(outerDimCount: outerDimCount)) / 255.0

    // Policy Network.
    let actionConv1 = relu(self.actionConv1(input))
    let actionConv2 = relu(self.actionConv2(actionConv1))
    let actionConv3 = relu(self.actionConv3(actionConv2)).reshaped(to: [-1, 3136])
    let actionHidden = relu(actionDenseHidden(actionConv3))
    let actionLogits = actionDenseOutput(actionHidden)
    let actionDistribution = Categorical<Int32>(logits: actionLogits)

    // Value Network.
    let valueConv1 = relu(self.valueConv1(input))
    let valueConv2 = relu(self.valueConv2(valueConv1))
    let valueConv3 = relu(self.valueConv3(valueConv2)).reshaped(to: [-1, 3136])
    let valueHidden = relu(valueDenseHidden(valueConv3))
    let value = valueDenseOutput(valueHidden)

    return ActorCriticOutput(
      actionDistribution: actionDistribution.unflattenedBatch(outerDims: outerDims),
      value: value.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
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
  parallelizedBatchProcessing: true)
let averageEpisodeReward = AverageEpisodeReward(for: arcadeEnvironment, bufferSize: 100)
let environment = EnvironmentCallbackWrapper(
  arcadeEnvironment,
  callbacks: averageEpisodeReward.updater())

logger.info("Number of valid actions: \(arcadeEnvironment.actionCount)")

let network = ArcadeActorCritic()
var agent = PPOAgent1(
  for: environment,
  network: network,
  optimizer: AMSGrad(for: network, learningRate: 2.5e-4),
  learningRateSchedule: ExponentialLearningRateDecay(decayRate: 0.99, decayStepCount: 3),
  maxGradientNorm: 0.5,
  advantageFunction: GeneralizedAdvantageEstimation(discountFactor: 0.99, discountWeight: 0.95),
  advantagesNormalizer: TensorNormalizer<Float>(streaming: true, alongAxes: 0, 1),
  useTDLambdaReturn: true,
  clip: PPOClip(epsilon: 0.1),
  penalty: PPOPenalty(klCutoffFactor: 0.5),
  valueEstimationLoss: PPOValueEstimationLoss(weight: 0.5, clipThreshold: nil),
  entropyRegularization: PPOEntropyRegularization(weight: 0.01),
  iterationCountPerUpdate: 4)
for step in 0..<6500 {
  let loss = agent.update(
    using: environment,
    maxSteps: 128 * batchSize,
    maxEpisodes: 128 * batchSize,
    stepCallbacks: [{ (environment, trajectory) in
      // if step > 0 { try! environment.render() }
    }])
  if step % 1 == 0 {
    logger.info("Step \(step) | Loss: \(loss) | Average Episode Reward: \(averageEpisodeReward.value())")
  }
}
