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

import ReinforcementLearning
import TensorFlow

public struct PPOAgent1<
  Environment: ReinforcementLearning.Environment,
  Network: Module,
  Optimizer: TensorFlow.Optimizer,
  LearningRate: ReinforcementLearning.LearningRate
>: PolicyGradientAgent
where
  Environment.Observation == Network.Input,
  Environment.ActionSpace.ValueDistribution: DifferentiableKLDivergence,
  Environment.Reward == Tensor<Float>,
  Network.Output == ActorCriticOutput<Environment.ActionSpace.ValueDistribution>,
  Optimizer.Model == Network,
  LearningRate.Scalar == Optimizer.Scalar
{
  public typealias Observation = Network.Input
  public typealias Action = ActionDistribution.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>

  public let actionSpace: Environment.ActionSpace
  public var network: Network
  public var optimizer: Optimizer
  public var trainingStep: UInt64 = 0

  public let learningRate: LearningRate
  public let maxGradientNorm: Float?
  public let advantageFunction: AdvantageFunction
  public let useTDLambdaReturn: Bool
  public let clip: PPOClip?
  public let penalty: PPOPenalty?
  public let valueEstimationLoss: PPOValueEstimationLoss
  public let entropyRegularization: PPOEntropyRegularization?
  public let iterationCountPerUpdate: Int

  @usableFromInline internal var advantagesNormalizer: TensorNormalizer<Float>?

  @inlinable
  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    learningRate: LearningRate,
    maxGradientNorm: Float? = 0.5,
    advantageFunction: AdvantageFunction = GeneralizedAdvantageEstimation(
      discountFactor: 0.99,
      discountWeight: 0.95),
    advantagesNormalizer: TensorNormalizer<Float> = TensorNormalizer<Float>(
      streaming: true,
      alongAxes: 0, 1),
    useTDLambdaReturn: Bool = true,
    clip: PPOClip? = PPOClip(),
    penalty: PPOPenalty? = PPOPenalty(),
    valueEstimationLoss: PPOValueEstimationLoss = PPOValueEstimationLoss(),
    entropyRegularization: PPOEntropyRegularization? = PPOEntropyRegularization(weight: 0.01),
    iterationCountPerUpdate: Int = 4
  ) {
    self.actionSpace = environment.actionSpace
    self.network = network
    self.optimizer = optimizer
    self.learningRate = learningRate
    self.maxGradientNorm = maxGradientNorm
    self.advantageFunction = advantageFunction
    self.advantagesNormalizer = advantagesNormalizer
    self.useTDLambdaReturn = useTDLambdaReturn
    self.clip = clip
    self.penalty = penalty
    self.valueEstimationLoss = valueEstimationLoss
    self.entropyRegularization = entropyRegularization
    self.iterationCountPerUpdate = iterationCountPerUpdate
  }

  @inlinable
  public func actionDistribution(for step: Step<Observation, Reward>) -> ActionDistribution {
    network(step.observation).actionDistribution
  }

  @inlinable
  @discardableResult
  public mutating func update(using trajectory: Trajectory<Observation, Action, Reward>) -> Float {
    optimizer.learningRate = learningRate(forStep: trainingStep)
    trainingStep += 1

    // Split the trajectory such that the last step is only used to provide the final value
    // estimate used for advantage estimation.
    let networkOutput = network(trajectory.observation)
    let sequenceLength = networkOutput.value.shape[0] - 1
    let stepKinds = StepKind(trajectory.stepKind.rawValue[0..<sequenceLength])
    let values = networkOutput.value[0..<sequenceLength]
    let finalValue = networkOutput.value[sequenceLength]

    // Estimate the advantages for the provided trajectory.
    let rewards = sign(trajectory.reward[0..<sequenceLength])
    let advantageEstimate = advantageFunction(
      stepKinds: stepKinds,
      rewards: rewards,
      values: values,
      finalValue: finalValue)
    var advantages = advantageEstimate.advantages
    let usingGAE = advantageFunction is GeneralizedAdvantageEstimation
    let returns = useTDLambdaReturn && usingGAE ?
      advantages + values :
      advantageEstimate.discountedReturns()
    advantagesNormalizer?.update(using: advantages)
    if let normalizer = advantagesNormalizer {
      advantages = normalizer.normalize(advantages)
    }

    // Compute the action log probabilities.
    let actionDistribution = networkOutput.actionDistribution
    let actionLogProbs = actionDistribution.logProbability(
      of: trajectory.action
    )[0..<sequenceLength]
    
    var lastEpochLoss: Float = 0.0
    for _ in 0..<iterationCountPerUpdate {
      var (loss, gradient) = network.valueWithGradient { network -> Tensor<Float> in
        let newNetworkOutput = network(trajectory.observation)

        // Compute the new action log probabilities.
        let newActionDistribution = newNetworkOutput.actionDistribution
        let newActionLogProbs = newActionDistribution.logProbability(
          of: trajectory.action
        )[0..<sequenceLength]

        let importanceRatio = exp(newActionLogProbs - actionLogProbs)
        var loss = importanceRatio * advantages

        // Importance ratio clipping loss term.
        if let c = self.clip {
          let ε = Tensor<Float>(c.epsilon)
          let importanceRatioClipped = importanceRatio.clipped(min: 1 - ε, max: 1 + ε)
          loss = -min(loss, importanceRatioClipped * advantages).mean()
        } else {
          loss = -loss.mean()
        }

        // KL penalty loss term.
        if let p = self.penalty {
          let klDivergence = actionDistribution.klDivergence(to: newActionDistribution)
          let klMean = klDivergence.mean()
          let klCutoffLoss = max(klMean - p.klCutoffFactor * p.adaptiveKLTarget, 0).squared()
          loss = loss + p.klCutoffCoefficient * klCutoffLoss
          if let beta = p.adaptiveKLBeta {
            loss = loss + beta * klMean
          }
        }

        // Entropy regularization loss term.
        if let e = self.entropyRegularization {
          let entropy = newActionDistribution.entropy()[0..<sequenceLength]
          loss = loss - e.weight * entropy.mean()
        }

        // Value estimation loss term.
        let newValues = newNetworkOutput.value[0..<sequenceLength]
        var valueLoss = (newValues - returns).squared()
        if let c = self.valueEstimationLoss.clipThreshold {
          let ε = Tensor<Float>(c)
          let clippedValues = values + (newValues - values).clipped(min: -ε, max: ε)
          let clippedValueLoss = (clippedValues - returns).squared()
          valueLoss = max(valueLoss, clippedValueLoss)
        }
        return loss + self.valueEstimationLoss.weight * valueLoss.mean()
      }
      if let clipNorm = maxGradientNorm {
        gradient.clipByGlobalNorm(clipNorm: clipNorm)
      }
      optimizer.update(&network, along: gradient)
      lastEpochLoss = loss.scalarized()
    }

    // After the network is updated, we may need to update the adaptive KL beta.
    if var p = penalty, let beta = p.adaptiveKLBeta {
      let klDivergence = network(trajectory.observation).actionDistribution.klDivergence(
        to: actionDistribution)
      let klMean = klDivergence.mean().scalarized()
      if klMean < p.adaptiveKLTarget / p.adaptiveKLToleranceFactor {
        p.adaptiveKLBeta = max(beta / p.adaptiveKLBetaScalingFactor, 1e-16)
      } else if klMean > p.adaptiveKLTarget * p.adaptiveKLToleranceFactor {
        p.adaptiveKLBeta = beta * p.adaptiveKLBetaScalingFactor
      }
    }

    return lastEpochLoss
  }
}
