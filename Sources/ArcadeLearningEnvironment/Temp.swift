import TensorFlow
import ReinforcementLearning

public struct PPOAgent1<
  Environment: ReinforcementLearning.Environment,
  Network: ReinforcementLearning.Network,
  Optimizer: TensorFlow.Optimizer
>: PolicyGradientAgent
where
  Environment.Observation == Network.Input,
  Environment.ActionSpace.ValueDistribution: DifferentiableKLDivergence,
  Environment.Reward == Tensor<Float>,
  Network.Output == ActorCriticOutput<Environment.ActionSpace.ValueDistribution>,
  Optimizer.Model == Network
{
  public typealias Observation = Network.Input
  public typealias Action = ActionDistribution.Value
  public typealias ActionDistribution = Environment.ActionSpace.ValueDistribution
  public typealias Reward = Tensor<Float>
  public typealias State = Network.State

  public let actionSpace: Environment.ActionSpace
  public var network: Network
  public var optimizer: Optimizer
  public var trainingStep: Int = 0

  public var state: State {
    get { network.state }
    set { network.state = newValue }
  }

  public let learningRateSchedule: LearningRateSchedule
  public let maxGradientNorm: Float?
  public let advantageFunction: AdvantageFunction
  public let useTDLambdaReturn: Bool
  public let clip: PPOClip?
  public let penalty: PPOPenalty?
  public let valueEstimationLoss: PPOValueEstimationLoss
  public let entropyRegularization: PPOEntropyRegularization?
  public let iterationCountPerUpdate: Int

  @usableFromInline internal var advantagesNormalizer: StreamingTensorNormalizer<Float>?

  @inlinable
  public init(
    for environment: Environment,
    network: Network,
    optimizer: Optimizer,
    learningRateSchedule: LearningRateSchedule,
    maxGradientNorm: Float? = 0.5,
    advantageFunction: AdvantageFunction = GeneralizedAdvantageEstimation(
      discountFactor: 0.99,
      discountWeight: 0.95),
    normalizeAdvantages: Bool = true,
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
    self.learningRateSchedule = learningRateSchedule
    self.maxGradientNorm = maxGradientNorm
    self.advantageFunction = advantageFunction
    self.advantagesNormalizer = normalizeAdvantages ?
      StreamingTensorNormalizer(alongAxes: 0, 1) :
      nil
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
  public mutating func update(
    using trajectory: Trajectory<Observation, Action, Reward, State>
  ) -> Float {
    optimizer.learningRate = learningRateSchedule.learningRate(step: trainingStep)
    trainingStep += 1

    network.state = trajectory.state
    let networkOutput = network(trajectory.observation)

    // Split the trajectory such that the last step is only used to provide the final value
    // estimate used for advantage estimation.
    let sequenceLength = networkOutput.value.shape[0] - 1
    let stepKinds = StepKind(trajectory.stepKind.rawValue[0..<sequenceLength])
    let values = networkOutput.value[0..<sequenceLength]
    let finalValue = networkOutput.value[sequenceLength]

    // Estimate the advantages for the provided trajectory.
    let rewards = trajectory.reward[0..<sequenceLength]
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
      // Restore the network state before computing the loss function.
      network.state = trajectory.state
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
        var valueEstimationLoss = (newValues - returns).squared()
        if let c = self.valueEstimationLoss.clipThreshold {
          let ε = Tensor<Float>(c)
          let clippedValues = values + (newValues - values).clipped(min: -ε, max: ε)
          let clippedValueEstimationLoss = (clippedValues - returns).squared()
          valueEstimationLoss = max(valueEstimationLoss, clippedValueEstimationLoss)
        }
        return loss + self.valueEstimationLoss.weight * valueEstimationLoss.mean()
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
