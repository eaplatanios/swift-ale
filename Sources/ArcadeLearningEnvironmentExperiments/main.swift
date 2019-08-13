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

let logger = Logger(label: "Breakout PPO")

let batchSize = 6
let emulators = (0..<batchSize).map { _ in ArcadeEmulator(game: .breakout) }
var environment = ArcadeEnvironment(
  using: emulators,
  observationsType: .screen(height: 84, width: 84, format: .grayscale),
  useMinimalActionSet: true,
  episodicLives: true,
  noOpReset: .stochastic(minCount: 0, maxCount: 30),
  frameSkip: .constant(4),
  frameStackCount: 4,
  parallelizedBatchProcessing: true)
var averageEpisodeReward = AverageEpisodeReward(for: environment, bufferSize: 100)

logger.info("Number of valid actions: \(environment.actionCount)")

struct ArcadeActorCritic: Module {
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
    outputSize: environment.actionCount,
    weightInitializer: orthogonal(gain: Tensor<Float>(0.01)))
  public var denseValue: Dense<Float> = Dense<Float>(
    inputSize: 512,
    outputSize: 1,
    weightInitializer: orthogonal(gain: Tensor<Float>(1.0)))

  @differentiable
  public func callAsFunction(_ input: Tensor<UInt8>) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.rank - 3
    let outerDims = [Int](input.shape.dimensions[0..<outerDimCount])
    let input = Tensor<Float>(input.flattenedBatch(outerDimCount: outerDimCount)) / 255.0

    // Policy Network.
    let conv1 = relu(self.conv1(input))
    let conv2 = relu(self.conv2(conv1))
    let conv3 = relu(self.conv3(conv2)).reshaped(to: [-1, 3136])
    let hidden = relu(denseHidden(conv3))
    let actionLogits = denseAction(hidden)
    let actionDistribution = Categorical<Int32>(logits: actionLogits)
    let value = denseValue(hidden)

    return ActorCriticOutput(
      actionDistribution: actionDistribution.unflattenedBatch(outerDims: outerDims),
      value: value.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}

let network = ArcadeActorCritic()
var agent = PPOAgent1(
  for: environment,
  network: network,
  optimizer: AMSGrad(for: network, learningRate: 1e-4),
  learningRateSchedule: ExponentialLearningRateDecay(decayRate: 0.999, decayStepCount: 1),
  maxGradientNorm: 0.5,
  advantageFunction: GeneralizedAdvantageEstimation(discountFactor: 0.99, discountWeight: 0.95),
  advantagesNormalizer: TensorNormalizer<Float>(streaming: false, alongAxes: 0, 1),
  useTDLambdaReturn: true,
  clip: PPOClip(epsilon: 0.1),
  penalty: nil,
  valueEstimationLoss: PPOValueEstimationLoss(weight: 0.5, clipThreshold: 0.1),
  entropyRegularization: PPOEntropyRegularization(weight: 0.01),
  iterationCountPerUpdate: 1)
for step in 0..<10000 {
  let loss = try! agent.update(
    using: &environment,
    maxSteps: 128 * batchSize,
    maxEpisodes: 128 * batchSize,
    callbacks: [{ (environment, trajectory) in
      averageEpisodeReward.update(using: trajectory)
      if step > 0 { try! environment.render() }
    }])
  if step % 1 == 0 {
    logger.info("Step \(step) | Loss: \(loss) | Average Episode Reward: \(averageEpisodeReward.value())")
  }
}
