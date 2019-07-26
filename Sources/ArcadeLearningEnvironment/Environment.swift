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

import Foundation
import ReinforcementLearning
import TensorFlow

/// Represents different settings for the observation space of the environment.
public enum ArcadeObservationsType {
  /// Use RGB image observations.
  case screen(height: Int, width: Int, format: ArcadeEmulator.ScreenFormat)

  /// Use RAM observations where you can see the memory of the game instead of the screen.
  case memory
}

public final class ArcadeEnvironment: RenderableEnvironment {
  public let batchSize: Int
  public let emulators: [ArcadeEmulator]
  public let useMinimalActionSet: Bool
  public let episodicLives: Bool
  public let noOpReset: NoOpReset
  public let frameSkip: FrameSkip
  public let frameStackCount: Int
  public let actionSpace: Discrete
  public let observationsType: ArcadeObservationsType
  public let observationSpace: DiscreteBox<UInt8>

  @usableFromInline internal let actionSet: [ArcadeEmulator.Action]
  @usableFromInline internal var currentLives: [Int]
  @usableFromInline internal var frameStacks: [FrameStack]
  @usableFromInline internal var needsReset: [Bool]
  @usableFromInline internal var rngs: [PhiloxRandomNumberGenerator]
  @usableFromInline internal var step: Step<Tensor<UInt8>, Tensor<Float>>? = nil
  @usableFromInline internal var renderer: ImageRenderer? = nil

  /// Dispatch queue used to synchronize updates on mutable shared objects when using
  /// parallelized batch processing.
  @usableFromInline internal let dispatchQueue: DispatchQueue?

  /// Number of distinct actions that are available in the managed arcade emulators.
  public var actionCount: Int { actionSet.count }

  @inlinable public var currentStep: Step<Tensor<UInt8>, Tensor<Float>> {
    if step == nil { step = reset() }
    return step!
  }

  public convenience init(
    using emulator: ArcadeEmulator,
    observationsType: ArcadeObservationsType = .screen(height: 84, width: 84, format: .grayscale),
    useMinimalActionSet: Bool = true,
    episodicLives: Bool = true,
    noOpReset: NoOpReset = .stochastic(minCount: 0, maxCount: 30),
    frameSkip: FrameSkip = .stochastic(minCount: 2, maxCount: 5),
    frameStackCount: Int = 4,
    renderer: ImageRenderer? = nil,
    parallelizedBatchProcessing: Bool = true
  ) {
    self.init(
      using: [emulator],
      observationsType: observationsType,
      useMinimalActionSet: useMinimalActionSet,
      episodicLives: episodicLives,
      noOpReset: noOpReset,
      frameSkip: frameSkip,
      frameStackCount: frameStackCount,
      renderer: renderer,
      parallelizedBatchProcessing: parallelizedBatchProcessing)
  }

  public init(
    using emulators: [ArcadeEmulator],
    observationsType: ArcadeObservationsType = .screen(height: 84, width: 84, format: .grayscale),
    useMinimalActionSet: Bool = true,
    episodicLives: Bool = true,
    noOpReset: NoOpReset = .stochastic(minCount: 0, maxCount: 30),
    frameSkip: FrameSkip = .stochastic(minCount: 2, maxCount: 5),
    frameStackCount: Int = 4,
    renderer: ImageRenderer? = nil,
    parallelizedBatchProcessing: Bool = true
  ) {
    precondition(emulators.count > 0, "At least one emulator must be provided.")
    precondition(frameStackCount > 1, "The frame stack size must be at least 1.")
    self.batchSize = emulators.count
    self.emulators = emulators
    self.useMinimalActionSet = useMinimalActionSet
    self.episodicLives = episodicLives
    self.noOpReset = noOpReset
    self.frameSkip = frameSkip
    self.frameStackCount = frameStackCount
    self.renderer = renderer
    self.dispatchQueue = parallelizedBatchProcessing && emulators.count > 1 ?
      DispatchQueue(label: "Arcade Learning Environment") :
      nil
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
    self.frameStacks = [FrameStack](repeating: FrameStack(size: frameStackCount), count: batchSize)
    self.needsReset = [Bool](repeating: true, count: batchSize)
    self.currentLives = emulators.map { $0.lives() }
    self.rngs = (0..<batchSize).map { _ in
      let seed = Context.local.randomSeed
      return PhiloxRandomNumberGenerator(seed: Int(seed.graph &+ seed.op))
    }
  }

  @inlinable
  @discardableResult
  public func step(taking action: Tensor<Int32>) -> Step<Tensor<UInt8>, Tensor<Float>> {
    let actions = action.unstacked()

    // Check if we need to use the parallelized version.
    if let dispatchQueue = self.dispatchQueue {
      var steps = [Step<Tensor<UInt8>, Tensor<Float>>?](repeating: nil, count: batchSize)
      DispatchQueue.concurrentPerform(iterations: batchSize) { batchIndex in
        if needsReset[batchIndex] {
          dispatchQueue.sync { steps[batchIndex] = reset(batchIndex: batchIndex) }
        }
        let action = actionSet[Int(actions[batchIndex].scalarized())]
        var reward = Float(0.0)
        let stepCount = dispatchQueue.sync { frameSkip.count(rng: &rngs[batchIndex]) }
        for _ in 0..<stepCount { reward += Float(emulators[batchIndex].act(using: action)) }
        let gameOver = emulators[batchIndex].gameOver()
        let lives = emulators[batchIndex].lives()
        let lostLife = lives < currentLives[batchIndex] && lives > 0
        let stepKind = gameOver ?
          StepKind.last() :
          episodicLives && lostLife ?
            StepKind.last(withReset: false) :
            StepKind.transition()
        let observation = dispatchQueue.sync { currentObservation(batchIndex: batchIndex) }
        let step = Step<Tensor<UInt8>, Tensor<Float>>(
          kind: stepKind,
          observation: observation,
          reward: Tensor<Float>(Float(reward)))
        dispatchQueue.sync {
          steps[batchIndex] = step
          needsReset[batchIndex] = gameOver
          if lostLife { currentLives[batchIndex] -= 1 }
        }
      }
      step = Step<Tensor<UInt8>, Tensor<Float>>.stack(steps.map { $0! })
      return step!
    }

    step = Step<Tensor<UInt8>, Tensor<Float>>.stack((0..<batchSize).map { batchIndex in
      if needsReset[batchIndex] { return reset(batchIndex: batchIndex) }
      let action = actionSet[Int(actions[batchIndex].scalarized())]
      var reward = Float(0.0)
      let stepCount = frameSkip.count(rng: &rngs[batchIndex])
      for _ in 0..<stepCount { reward += Float(emulators[batchIndex].act(using: action)) }
      let gameOver = emulators[batchIndex].gameOver()
      let lives = emulators[batchIndex].lives()
      let lostLife = lives < currentLives[batchIndex] && lives > 0
      let stepKind = gameOver ?
        StepKind.last() :
        episodicLives && lostLife ?
          StepKind.last(withReset: false) :
          StepKind.transition()
      needsReset[batchIndex] = gameOver
      if lostLife { currentLives[batchIndex] -= 1 }
      return Step<Tensor<UInt8>, Tensor<Float>>(
        kind: stepKind,
        observation: currentObservation(batchIndex: batchIndex),
        reward: Tensor<Float>(Float(reward)))
    })
    return step!
  }

  @inlinable
  @discardableResult
  public func reset() -> Step<Tensor<UInt8>, Tensor<Float>> {
    step = Step<Tensor<UInt8>, Tensor<Float>>.stack((0..<batchSize).map { reset(batchIndex: $0) })
    return step!
  }

  @inlinable
  @discardableResult
  public func reset(batchIndex: Int) -> Step<Tensor<UInt8>, Tensor<Float>> {
    emulators[batchIndex].resetGame()
    for _ in 0..<noOpReset.count(rng: &rngs[batchIndex]) {
      emulators[batchIndex].act(using: .noOp)
      if emulators[batchIndex].gameOver() {
        emulators[batchIndex].resetGame()
      }
    }
    needsReset[batchIndex] = false
    currentLives[batchIndex] = emulators[batchIndex].lives()
    return Step<Tensor<UInt8>, Tensor<Float>>(
      kind: .first(),
      observation: currentObservation(batchIndex: batchIndex),
      reward: Tensor<Float>(0.0))
  }

  @inlinable
  public func copy() -> ArcadeEnvironment {
    ArcadeEnvironment(
      using: emulators.map { ArcadeEmulator(copying: $0) },
      observationsType: observationsType,
      useMinimalActionSet: useMinimalActionSet,
      episodicLives: episodicLives,
      frameSkip: frameSkip,
      frameStackCount: frameStackCount,
      renderer: renderer,
      parallelizedBatchProcessing: dispatchQueue != nil)
  }

  @inlinable
  public func render() throws {
    if renderer == nil { renderer = ImageRenderer() }
    // TODO: Better support batchSize > 1.
    try renderer!.render(emulators[0].screen(format: .rgb).array)
  }

  @inlinable
  internal func currentObservation(batchIndex: Int) -> Tensor<UInt8> {
    var frame: Tensor<UInt8>
    switch observationsType {
    case let .screen(height, width, format):
      let emulatorScreen = emulators[batchIndex].screen(format: format)
      frame = resize(
        images: emulatorScreen,
        to: Tensor<Int32>([Int32(height), Int32(width)]),
        method: .area)
    case .memory:
      frame = emulators[batchIndex].memory()
    }
    frameStacks[batchIndex].push(frame)
    return Tensor<UInt8>(concatenating: frameStacks[batchIndex].frames(), alongAxis: -1)
  }
}

extension ArcadeEnvironment {
  public enum NoOpReset {
    case none
    case constant(_ count: Int)
    case stochastic(minCount: Int, maxCount: Int)

    @inlinable
    internal func count<G: RandomNumberGenerator>(rng: inout G) -> Int {
      switch self {
      case .none: return 0
      case let .constant(count): return count
      case let .stochastic(min, max): return Int.random(in: min...max, using: &rng)
      }
    }
  }

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

  // TODO: Find a way to make this "replay-buffer-friendly" so that it doesn't store duplicate
  // copies of each frame.
  public struct FrameStack {
    public let size: Int

    @usableFromInline internal var index: Int = 0
    @usableFromInline internal var buffer: [Tensor<UInt8>] = []

    @inlinable
    public init(size: Int) {
      self.size = size
    }

    @inlinable
    public mutating func push(_ frame: Tensor<UInt8>) {
      if buffer.count == size {
        buffer[index] = frame
      } else {
        buffer.append(frame)
      }
      index = (index + 1) % size
    }

    @inlinable
    public func frames() -> [Tensor<UInt8>] {
      if buffer.count == size {
        if index == 0 { return buffer }
        return [Tensor<UInt8>](buffer[(index - 1)...] + buffer[..<(index - 1)])
      } else {
        return [Tensor<UInt8>](
          [Tensor<UInt8>](repeating: buffer[0], count: size - index) + buffer[0..<index])
      }
    }
  }
}
