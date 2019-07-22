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

/// Represents different settings for the observation space of the environment.
public enum ArcadeObservationsType {
  /// Use RGB image observations.
  case screen(height: Int, width: Int, format: ArcadeEmulator.ScreenFormat)

  /// Use RAM observations where you can see the memory of the game instead of the screen.
  case memory
}

public struct ArcadeEnvironment: Environment {
  public let batchSize: Int
  public let emulators: [ArcadeEmulator]
  public let useMinimalActionSet: Bool
  public let frameSkip: FrameSkip
  public let actionSpace: Discrete
  public let observationsType: ArcadeObservationsType
  public let observationSpace: DiscreteBox<UInt8>

  @usableFromInline internal let actionSet: [ArcadeEmulator.Action]
  @usableFromInline internal var needsReset: [Bool]
  @usableFromInline internal var rngs: [PhiloxRandomNumberGenerator]
  @usableFromInline internal var step: Step<Tensor<UInt8>, Tensor<Float>>? = nil

  /// Number of distinct actions that are available in the managed arcade emulators.
  public var actionCount: Int { actionSet.count }

  public init(
    using emulator: ArcadeEmulator,
    observationsType: ArcadeObservationsType = .screen(height: 84, width: 84, format: .grayscale),
    useMinimalActionSet: Bool = true,
    frameSkip: FrameSkip = .stochastic(minCount: 2, maxCount: 5)
  ) {
    self.init(
      using: [emulator],
      observationsType: observationsType,
      useMinimalActionSet: useMinimalActionSet)
  }

  public init(
    using emulators: [ArcadeEmulator],
    observationsType: ArcadeObservationsType = .screen(height: 84, width: 84, format: .grayscale),
    useMinimalActionSet: Bool = true,
    frameSkip: FrameSkip = .stochastic(minCount: 2, maxCount: 5)
  ) {
    precondition(emulators.count > 0, "At least one emulator must be provided.")
    self.batchSize = emulators.count
    self.emulators = emulators
    self.useMinimalActionSet = useMinimalActionSet
    self.frameSkip = frameSkip
    self.actionSet = useMinimalActionSet ?
        emulators[0].minimalActions() :
        emulators[0].legalActions()
    self.actionSpace = Discrete(withSize: actionSet.count, batchSize: batchSize)
    self.observationsType = observationsType
    switch observationsType {
    case let .screen(height, width, format):
      self.observationSpace = DiscreteBox<UInt8>(
        shape: format.shape(height: height, width: width),
        lowerBound: 0,
        upperBound: 255)
    case .memory:
      self.observationSpace = DiscreteBox<UInt8>(
        shape: TensorShape(emulators[0].memorySize()),
        lowerBound: 0,
        upperBound: 255)
    }
    self.needsReset = [Bool](repeating: true, count: batchSize)
    self.rngs = (0..<batchSize).map { _ in
      let seed = Context.local.randomSeed
      return PhiloxRandomNumberGenerator(seed: Int(seed.graph + seed.op))
    }
  }

  @inlinable
  public mutating func currentStep() -> Step<Tensor<UInt8>, Tensor<Float>> {
    if step == nil { step = reset() }
    return step!
  }

  @inlinable
  internal func currentObservation(batchIndex: Int) -> Tensor<UInt8> {
    switch observationsType {
    case let .screen(height, width, format):
      let emulatorScreen = emulators[batchIndex].screen(format: format)
      return resize(
        images: emulatorScreen,
        to: Tensor<Int32>([Int32(height), Int32(width)]),
        method: .area)
    case .memory:
      return emulators[batchIndex].memory()
    }
  }

  @discardableResult
  @inlinable
  public mutating func step(taking action: Tensor<Int32>) -> Step<Tensor<UInt8>, Tensor<Float>> {
    let actions = action.unstacked()
    step = Step<Tensor<UInt8>, Tensor<Float>>.stack((0..<batchSize).map {
      step(taking: actions[$0], batchIndex: $0)
    })
    return step!
  }

  @discardableResult
  @inlinable
  public mutating func step(
    taking action: Tensor<Int32>,
    batchIndex: Int
  ) -> Step<Tensor<UInt8>, Tensor<Float>> {
    if needsReset[batchIndex] { reset(batchIndex: batchIndex) }
    let action = actionSet[Int(action.scalarized())]
    var reward = Float(0.0)
    let stepCount = frameSkip.count(rng: &rngs[batchIndex])
    for _ in 0..<stepCount { reward += Float(emulators[batchIndex].act(using: action)) }   
    let finished = emulators[batchIndex].gameOver()
    if finished { needsReset[batchIndex] = true }
    return Step<Tensor<UInt8>, Tensor<Float>>(
      kind: finished ? .last : .transition,
      observation: currentObservation(batchIndex: batchIndex),
      reward: Tensor<Float>(Float(reward)))
  }

  @discardableResult
  @inlinable
  public mutating func reset() -> Step<Tensor<UInt8>, Tensor<Float>> {
    step = Step<Tensor<UInt8>, Tensor<Float>>.stack((0..<batchSize).map { reset(batchIndex: $0) })
    return step!
  }

  @discardableResult
  @inlinable
  public mutating func reset(batchIndex: Int) -> Step<Tensor<UInt8>, Tensor<Float>> {
    emulators[batchIndex].resetGame()
    needsReset[batchIndex] = false
    return Step<Tensor<UInt8>, Tensor<Float>>(
      kind: .first,
      observation: currentObservation(batchIndex: batchIndex),
      reward: Tensor<Float>(0.0))
  }

  @inlinable
  public func copy() -> ArcadeEnvironment {
    ArcadeEnvironment(
      using: emulators.map { ArcadeEmulator(copying: $0) },
      observationsType: observationsType,
      useMinimalActionSet: useMinimalActionSet,
      frameSkip: frameSkip)
  }
}

extension ArcadeEnvironment {
  public enum FrameSkip {
    case none
    case constant(_ count: Int)
    case stochastic(minCount: Int, maxCount: Int)

    @inlinable
    internal func count<G: RandomNumberGenerator>(rng: inout G) -> Int {
      switch self {
      case .none: return 1
      case let .constant(count): return count
      case let .stochastic(min, max): return Int.random(in: min...max, using: &rng)
      }
    }
  }
}
