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

/// Observations type for arcade environments.
public enum ArcadeObservationsType {
  /// Use RGB image observations.
  case screen(height: Int, width: Int, format: ArcadeEmulator.ScreenFormat)

  /// Use RAM observations where you can see the memory of the game instead of the screen.
  case memory
}

/// Atari game environment.
public final class ArcadeEnvironment: RenderableEnvironment {
  /// Batch size of this environment.
  public let batchSize: Int

  /// Arcade emulators used by this environment for Atari game emulation.
  public let emulators: [ArcadeEmulator]

  /// Indicates whether to use the minimal valid action set instead of the full Atari action set.
  public let useMinimalActionSet: Bool

  /// Indicates whether loss of life in a game corresponds to an episode ending. If `false`, game
  /// over corresponds to an episode ending, which may involve multiple lost lives.
  public let episodicLives: Bool

  /// No-op reset configuration used when resetting each game.
  public let noOpReset: NoOpReset

  /// Frame skipping strategy used for each simulation step.
  public let frameSkip: FrameSkip

  /// Number of frames that are stacked to form each observation. The stacked frames are always the
  /// most recent `frameStackCount` frames.
  public let frameStackCount: Int

  /// Indicates whether the batched emulators are running in parallel.
  public let parallelizedBatchProcessing: Bool

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

  /// Dispatch queue used to synchronize updates on mutable shared objects when using parallelized
  /// batch processing.
  @usableFromInline internal let dispatchQueue: DispatchQueue =
    DispatchQueue(label: "Arcade Learning Environment")

  /// Number of distinct actions that are available in the managed arcade emulators.
  public var actionCount: Int { actionSet.count }

  /// Current environment step (i.e., result of the last call to `ArcadeEnvironment.step()`).
  @inlinable public var currentStep: Step<Tensor<UInt8>, Tensor<Float>> {
    if step == nil { step = reset() }
    return step!
  }

  public init(
    using emulators: [ArcadeEmulator],
    observationsType: ArcadeObservationsType = .screen(height: 84, width: 84, format: .grayscale),
    useMinimalActionSet: Bool = false,
    episodicLives: Bool = true,
    noOpReset: NoOpReset = .stochastic(minCount: 0, maxCount: 30),
    frameSkip: FrameSkip = .stochastic(minCount: 2, maxCount: 5),
    frameStackCount: Int = 4,
    parallelizedBatchProcessing: Bool = true,
    renderer: ImageRenderer? = nil
  ) {
    precondition(emulators.count > 0, "At least one emulator must be provided.")
    precondition(frameStackCount > 0, "The frame stack size must be at least 1.")
    precondition(
      !useMinimalActionSet || emulators.allSatisfy { $0.game == emulators.first!.game },
      "If `useMinimalActionSet` is set, all provided emulators must be emulating the same game.")
    self.batchSize = emulators.count
    self.emulators = emulators
    self.useMinimalActionSet = useMinimalActionSet
    self.episodicLives = episodicLives
    self.noOpReset = noOpReset
    self.frameSkip = frameSkip
    self.frameStackCount = frameStackCount
    self.parallelizedBatchProcessing = parallelizedBatchProcessing
    self.renderer = renderer
    self.actionSet = useMinimalActionSet ?
        emulators[0].minimalActions :
        emulators[0].legalActions
    self.actionSpace = Discrete(withSize: actionSet.count, batchSize: batchSize)
    self.observationsType = observationsType
    switch observationsType {
    case let .screen(height, width, format):
      self.observationSpace = DiscreteBox<UInt8>(
        shape: TensorShape(format.shape(height: height, width: width)),
        lowerBound: 0,
        upperBound: 255)
    case .memory:
      self.observationSpace = DiscreteBox<UInt8>(
        shape: TensorShape(emulators[0].memorySize()),
        lowerBound: 0,
        upperBound: 255)
    }
    self.frameStacks = frameStackCount > 1 ?
      [FrameStack](repeating: FrameStack(size: frameStackCount), count: batchSize) :
      [FrameStack]()
    self.needsReset = [Bool](repeating: true, count: batchSize)
    self.currentLives = emulators.map { $0.lives }
    self.rngs = (0..<batchSize).map { _ in
      let seed = Context.local.randomSeed
      return PhiloxRandomNumberGenerator(seed: Int(seed.graph &+ seed.op))
    }
  }

  /// States corresponding to the underlying Atari emulators.
  /// - Note: These states do *not* include pseudorandomness, making them suitable for planning
  ///   purposes. In contrast, see `ArcadeEnvironment.systemStates`.
  @inlinable
  public var states: [[Int8]] {
    get { emulators.map { $0.state.encoded() } }
    set {
      let states = newValue.map { ArcadeEmulator.State(decoding: $0) }
      for i in 0..<batchSize {
        emulators[i].state = states[i]
      }
    }
  }

  /// States corresponding to the underlying Atari emulators that are suitable for serialization.
  /// - Note: These states include pseudorandomness, making them unsuitable for planning purposes.
  ///   In contrast, see `ArcadeEnvironment.states`.
  @inlinable
  public var systemStates: [[Int8]] {
    get { emulators.map { $0.systemState.encoded() } }
    set {
      let systemStates = newValue.map { ArcadeEmulator.State(decoding: $0) }
      for i in 0..<batchSize {
        emulators[i].systemState = systemStates[i]
      }
    }
  }

  /// Performs a step in this environment using the provided action and returns information about
  /// the performed step.
  @inlinable
  @discardableResult
  public func step(taking action: Tensor<Int32>) -> Step<Tensor<UInt8>, Tensor<Float>> {
    let actions = action.unstacked()

    // Check if we need to use the parallelized version.
    if parallelizedBatchProcessing {
      var steps = [Step<Tensor<UInt8>, Tensor<Float>>?](repeating: nil, count: batchSize)
      DispatchQueue.concurrentPerform(iterations: batchSize) { batchIndex in
        let step = self.step(taking: actions[batchIndex], batchIndex: batchIndex)
        dispatchQueue.sync { steps[batchIndex] = step }
      }
      step = Step<Tensor<UInt8>, Tensor<Float>>.stack(steps.map { $0! })
      return step!
    }

    step = Step<Tensor<UInt8>, Tensor<Float>>.stack((0..<batchSize).map { batchIndex in
      self.step(taking: actions[batchIndex], batchIndex: batchIndex)
    })
    return step!
  }

  /// Performs a step in this environment for the specified batch index, using the provided action,
  /// and returns information about the performed step.
  @inlinable
  internal func step(
    taking action: Tensor<Int32>,
    batchIndex: Int
  ) -> Step<Tensor<UInt8>, Tensor<Float>> {
    if needsReset[batchIndex] { return reset(batchIndex: batchIndex) }
    let action = actionSet[Int(action.scalarized())]
    let stepCount = dispatchQueue.sync { frameSkip.count(rng: &rngs[batchIndex]) }
    var reward = Float(0.0)
    var observations = [Tensor<UInt8>]()
    for _ in 0..<stepCount {
      reward += Float(emulators[batchIndex].act(using: action))
      observations.append(currentObservation(batchIndex: batchIndex))
    }
    let observation = stepCount > 1 ?
      Tensor<UInt8>(stacking: observations, alongAxis: -1).max(squeezingAxes: -1) :
      observations[0]
    let gameOver = emulators[batchIndex].gameOver
    let lives = emulators[batchIndex].lives
    let lostLife = lives < currentLives[batchIndex] && lives > 0
    let stepKind = gameOver ?
      StepKind.last() :
      episodicLives && lostLife ?
        StepKind.last(withReset: false) :
        StepKind.transition()
    dispatchQueue.sync {
      needsReset[batchIndex] = gameOver
      if lostLife { currentLives[batchIndex] -= 1 }
    }
    return Step<Tensor<UInt8>, Tensor<Float>>(
      kind: stepKind,
      observation: observation,
      reward: Tensor<Float>(Float(reward)))
  }

  /// Resets this environment and returns the first step.
  @inlinable
  @discardableResult
  public func reset() -> Step<Tensor<UInt8>, Tensor<Float>> {
    step = Step<Tensor<UInt8>, Tensor<Float>>.stack((0..<batchSize).map { reset(batchIndex: $0) })
    return step!
  }

  /// Resets this environment for the specified batch index and returns the first step of the
  /// resetted environment for that batch index.
  @inlinable
  internal func reset(batchIndex: Int) -> Step<Tensor<UInt8>, Tensor<Float>> {
    emulators[batchIndex].resetGame()
    for _ in 0..<noOpReset.count(rng: &rngs[batchIndex]) {
      emulators[batchIndex].act(using: .noOp)
      if emulators[batchIndex].gameOver {
        emulators[batchIndex].resetGame()
      }
    }
    needsReset[batchIndex] = false
    currentLives[batchIndex] = emulators[batchIndex].lives
    return Step<Tensor<UInt8>, Tensor<Float>>(
      kind: .first(),
      observation: currentObservation(batchIndex: batchIndex),
      reward: Tensor<Float>(0.0))
  }

  /// Returns a copy of this environment.
  @inlinable
  public func copy() -> ArcadeEnvironment {
    ArcadeEnvironment(
      using: emulators.map { ArcadeEmulator(copying: $0) },
      observationsType: observationsType,
      useMinimalActionSet: useMinimalActionSet,
      episodicLives: episodicLives,
      frameSkip: frameSkip,
      frameStackCount: frameStackCount,
      parallelizedBatchProcessing: parallelizedBatchProcessing,
      renderer: renderer)
  }

  /// Renders the current state of this environment.
  @inlinable
  public func render() throws {
    if renderer == nil { renderer = ImageRenderer() }
    // TODO: Better support batchSize > 1.
    try renderer!.render(emulators[0].screen(format: .rgb))
  }

  /// Returns the current obervation from the emulator corresponding to `batchIndex`.
  @inlinable
  internal func currentObservation(batchIndex: Int) -> Tensor<UInt8> {
    let emulator = dispatchQueue.sync { emulators[batchIndex] }
    var frame: ShapedArray<UInt8>
    switch observationsType {
    case let .screen(height, width, format):
      let emulatorScreen = emulator.screen(format: format)
      frame = resize(
        images: Tensor(emulatorScreen),
        to: Tensor<Int32>([Int32(height), Int32(width)]),
        method: .area).array
    case .memory:
      frame = ShapedArray(emulator.memory())
    }
    if frameStacks.count > 0 {
      dispatchQueue.sync { frameStacks[batchIndex].push(frame) }
      let frames = frameStacks[batchIndex].frames().map(Tensor.init)
      return Tensor<UInt8>(concatenating: frames, alongAxis: -1)
    }
    return Tensor<UInt8>(frame)
  }
}

extension ArcadeEnvironment {
  /// No-op reset configuration used when resetting each game.
  public enum NoOpReset {
    /// No no-op actions are performed.
    case none

    /// A fixed number of `count` no-op actions are performed after each game reset.
    case constant(_ count: Int)

    /// A random number of between `minCount` and `maxCount` no-op actions are performed after
    /// each game reset.
    case stochastic(minCount: Int, maxCount: Int)

    /// Returns the number of no-op actions to take after resetting a game.
    @inlinable
    internal func count<G: RandomNumberGenerator>(rng: inout G) -> Int {
      switch self {
      case .none: return 0
      case let .constant(count): return count
      case let .stochastic(min, max): return Int.random(in: min...max, using: &rng)
      }
    }
  }

  /// Frame skipping strategy used for each simulation step.
  public enum FrameSkip {
    /// No frame skipping is performed.
    case none

    /// A fixed number of `count` frames are skipped after each environment step. Frame skipping is
    /// performed by repeating the same action a specified number of times.
    case constant(_ count: Int)

    /// A random number of between `minCount` and `maxCount` frames are skipped after each
    /// environment step. Frame skipping is performed by repeating the same action a specified
    /// number of times.
    case stochastic(minCount: Int, maxCount: Int)

    /// Returns the number of frames to skip after each simulation step.
    @inlinable
    internal func count<G: RandomNumberGenerator>(rng: inout G) -> Int {
      switch self {
      case .none: return 1
      case let .constant(count): return count
      case let .stochastic(min, max): return Int.random(in: min...max, using: &rng)
      }
    }
  }

  @usableFromInline
  internal struct FrameStack {
    @usableFromInline internal let size: Int
    @usableFromInline internal var index: Int = 0
    @usableFromInline internal var buffer: [ShapedArray<UInt8>] = []

    // TODO: Find a way to make this "replay-buffer-friendly" so that it doesn't store duplicate
    // copies of each frame.

    @inlinable
    internal init(size: Int) {
      self.size = size
    }

    @inlinable
    internal mutating func push(_ frame: ShapedArray<UInt8>) {
      if buffer.count == size {
        buffer[index] = frame
      } else {
        buffer.append(frame)
      }
      index = (index + 1) % size
    }

    @inlinable
    internal func frames() -> [ShapedArray<UInt8>] {
      if buffer.count == size {
        if index == 0 { return buffer }
        return [ShapedArray<UInt8>](buffer[(index - 1)...] + buffer[..<(index - 1)])
      } else {
        return [ShapedArray<UInt8>](
          [ShapedArray<UInt8>](repeating: buffer[0], count: size - index) + buffer[0..<index])
      }
    }
  }
}
