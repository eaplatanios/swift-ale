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

import CArcadeLearningEnvironment
import Foundation
import TensorFlow

/// Arcade games emulator, based on the Arcade Learning Environment (ALE).
public final class ArcadeEmulator {
  /// Path to the folder in which the game ROMs are stored.
  public let gameROMsPath: URL

  /// Indicates whether color averaging is enabled. Many Atari 2600 games display objects on
  /// alternating frames (and sometimes even less frequently). This can be an issue for agents that
  /// do not consider the whole screen history. If color averaging is enabled, the emulator output
  /// (as observed by the agents) is a weighted blend of the last two frames.
  public let colorAveraging: Bool

  /// Probability of repeating the previous action, instead of the one requested by the agent. With
  /// probability `repeatActionProbability`, the previously executed action is executed again
  /// during the next frame, ignoring the agent's actual choice. The default value was chosen as
  /// the highest value for which human players were unable to detect any delay or control lag. The
  /// motivation for introducing action repeat stochasticity was to help separate trajectory
  /// optimization research from robust controller optimization, the latter often being the desired
  /// outcome in reinforcement learning (RL). We strongly encourage RL researchers to use the
  /// default stochasticity level in their agents, and clearly report the setting used.
  public let repeatActionProbability: Float

  /// Initial random seed used by this emulator.
  public let randomSeed: TensorFlowSeed?

  /// Pointer to the underlying native library emulator instance.
  @usableFromInline internal var handle: UnsafeMutablePointer<ALEInterface?>?

  /// Current game loaded in this emulator.
  @usableFromInline internal var currentGame: Game

  /// Current logging mode of this emulator.
  @usableFromInline internal var currentLoggingMode: LoggingMode = .error

  /// Current game mode.
  @usableFromInline internal var currentGameMode: Int = 0

  /// Current game difficulty.
  @usableFromInline internal var currentGameDifficulty: Int = 0

  /// Creates a new arcade emulator.
  ///
  /// - Parameters:
  ///   - game: Game to load in the new emulator.
  ///   - gameMode: Game mode to use.
  ///   - gameDifficulty: Game difficulty level to use.
  ///   - gameROMsPath: Path to the folder in which the game ROMs are stored.
  ///   - colorAveraging: Indicates whether color averaging is enabled. Many Atari 2600 games
  ///     display objects on alternating frames (and sometimes even less frequently). This can be
  ///     an issue for agents that do not consider the whole screen history. If color averaging is
  ///     enabled, the emulator output (as observed by the agents) is a weighted blend of the last
  ///     two frames.
  ///   - repeatActionProbability: Probability of repeating the previous action, instead of the one
  ///     requested by the agent. With probability `repeatActionProbability`, the previously
  ///     executed action is executed again during the next frame, ignoring the agent's actual
  ///     choice. The default value was chosen as the highest value for which human players were
  ///     unable to detect any delay or control lag. The motivation for introducing action repeat
  ///     stochasticity was to help separate trajectory optimization research from robust
  ///     controller optimization, the latter often being the desired outcome in reinforcement
  ///     learning (RL). We strongly encourage RL researchers to use the default stochasticity
  ///     level in their agents, and clearly report the setting used.
  ///   - loggingMode: Logging mode for the new emulator.
  ///   - randomSeed: Initial random seed to use for the new emulator.
  ///
  /// - Note: If the game ROM cannot be found in `gameROMsPath`, an attempt will be made to
  ///   download it in that folder.
  @inlinable
  public init(
    game: Game,
    gameMode: Int = 0,
    gameDifficulty: Int = 0,
    gameROMsPath: URL? = nil,
    colorAveraging: Bool = true,
    repeatActionProbability: Float = 0.25,
    loggingMode: LoggingMode = .error,
    randomSeed: TensorFlowSeed? = nil
  ) {
    setLoggerMode(loggingMode.rawValue)
    self.currentGame = game
    self.handle = ALE_new()
    self.gameROMsPath = gameROMsPath ?? 
      URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
      .appendingPathComponent("data")
      .appendingPathComponent("roms")
    self.colorAveraging = colorAveraging
    self.repeatActionProbability = repeatActionProbability
    self.randomSeed = randomSeed
    if let r = randomSeed { self["random_seed"] = Int(r.graph &+ r.op) }
    self["color_averaging"] = colorAveraging
    self["repeat_action_probability"] = repeatActionProbability
    self.game = game
    self.loggingMode = loggingMode
    self.gameMode = gameMode
    self.gameDifficulty = gameDifficulty
  }

  /// Creates a new arcade emulator by copying another emulator.
  @inlinable
  public convenience init(copying emulator: ArcadeEmulator) {
    self.init(
      game: emulator.game,
      gameMode: emulator.gameMode,
      gameDifficulty: emulator.gameDifficulty,
      gameROMsPath: emulator.gameROMsPath,
      colorAveraging: emulator.colorAveraging,
      repeatActionProbability: emulator.repeatActionProbability,
      loggingMode: emulator.loggingMode,
      randomSeed: emulator.randomSeed)
  }

  @inlinable
  deinit {
    if let h = handle { ALE_del(h) }
  }

  /// Game loaded in this emulator.
  @inlinable
  public var game: Game {
    get { currentGame }
    set {
      try! loadROM(handle, newValue.romPath(in: gameROMsPath).path)
      currentGame = newValue
    }
  }

  /// Logging mode.
  @inlinable
  public var loggingMode: LoggingMode {
    get { currentLoggingMode }
    set {
      setLoggerMode(newValue.rawValue)
      currentLoggingMode = newValue
    }
  }

  /// Indicates whether the game has ended.
  @inlinable
  public var gameOver: Bool { game_over(handle) }

  /// Number of lives left in the current game. If the current game does not have a concept of
  /// lives (e.g., Freeway), then `lives` is set to `0`.
  @inlinable
  public var lives: Int { Int(CArcadeLearningEnvironment.lives(handle)) }

  /// Frame number since loading the game ROM.
  @inlinable
  public var frameNumber: Int { Int(getFrameNumber(handle)) }

  /// Frame number since the start of the current episode.
  @inlinable
  public var episodeFrameNumber: Int { Int(getEpisodeFrameNumber(handle)) }

  /// Game mode.
  @inlinable
  public var gameMode: Int {
    get { currentGameMode }
    set {
      CArcadeLearningEnvironment.setMode(handle, Int32(newValue))
      currentGameMode = newValue
    }
  }

  /// Game modes supported by the current game.
  @inlinable
  public var supportedGameModes: [Int] {
    let count = Int(getAvailableModesSize(handle))
    let modes = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    defer { modes.deallocate() }
    getAvailableModes(handle, modes)
    return [Int32](UnsafeBufferPointer(start: modes, count: count)).map(Int.init)
  }

  /// Game difficulty.
  @inlinable
  public var gameDifficulty: Int {
    get { currentGameDifficulty }
    set {
      CArcadeLearningEnvironment.setDifficulty(handle, Int32(newValue))
      currentGameDifficulty = newValue
    }
  }

  /// Game difficulty levels supported by the current game.
  @inlinable
  public var supportedGameDifficulties: [Int] {
    let count = Int(getAvailableDifficultiesSize(handle))
    let difficulties = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    defer { difficulties.deallocate() }
    getAvailableDifficulties(handle, difficulties)
    return [Int32](UnsafeBufferPointer(start: difficulties, count: count)).map(Int.init)
  }

  /// All possible actions supported by this emulator.
  @inlinable
  public var legalActions: [Action] {
    let count = Int(getLegalActionSize(handle))
    let actions = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    defer { actions.deallocate() }
    getLegalActionSet(handle, actions)
    return [Int32](UnsafeBufferPointer(start: actions, count: count)).map { Action(rawValue: $0)! }
  }

  /// Minimal set of actions needed to play the current game (i.e., all of the returned actions
  /// have some effect in the game).
  @inlinable
  public var minimalActions: [Action] {
    let count = Int(getMinimalActionSize(handle))
    let actions = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    defer { actions.deallocate() }
    getMinimalActionSet(handle, actions)
    return [Int32](UnsafeBufferPointer(start: actions, count: count)).map { Action(rawValue: $0)! }
  }

  /// State of this emulator.
  /// - Note: This state does *not* include pseudorandomness, making it suitable for planning
  ///   purposes. In contrast, see `Emulator.systemState`.
  @inlinable
  public var state: State {
    get { State(handle: cloneState(handle)) }
    set { restoreState(handle, newValue.handle) }
  }

  /// State of this emulator that is suitable for serialization.
  /// - Note: This state includes pseudorandomness, making it unsuitable for planning purposes.
  ///   In contrast, see `Emulator.state`.
  @inlinable
  public var systemState: State {
    get { State(handle: cloneSystemState(handle)) }
    set { restoreSystemState(handle, newValue.handle) }
  }

  /// Resets the game, but not the full system (i.e., this is not "equivalent" to unplugging the
  /// console from electricity).
  @inlinable
  public func resetGame() {
    reset_game(handle)
  }

  /// Applies the provided action to the game and returns the obtained reward. It is the user's
  /// responsibility to check if the game has ended and to reset it when necessary (this method
  /// will keep pressing buttons on the game over screen).
  @inlinable
  @discardableResult
  public func act(using action: Action) -> Int {
    Int(CArcadeLearningEnvironment.act(handle, action.rawValue))
  }

  /// Returns the screen size of this emulator.
  @inlinable
  public func screenSize() -> (height: Int, width: Int) {
    (height: Int(getScreenHeight(handle)), width: Int(getScreenWidth(handle)))
  }

  /// Returns a shaped array filled with the pixel data from this emulator's screen.
  ///
  /// - Parameter format: Screen format to use (e.g., RGB pixel values).
  /// - Returns: Shaped array with shape determined by the requested `format`.
  @inlinable
  public func screen(format: ScreenFormat = .rgb) -> ShapedArray<UInt8> {
    let (height, width) = screenSize()
    let size = format.size(height: height, width: width)
    let screen = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
    defer { screen.deallocate() }
    switch format {
    case .raw: getScreen(handle, screen)
    case .rgb: getScreenRGB(handle, screen)
    case .grayscale: getScreenGrayscale(handle, screen)
    }
    let screenArray = [UInt8](UnsafeBufferPointer(start: screen, count: size))
    return ShapedArray(shape: format.shape(height: height, width: width), scalars: screenArray)
  }

  /// Saves the a screenshot of the emulator in the provided path, in PNG format.
  @inlinable
  public func saveScreen(in path: String) {
    saveScreenPNG(handle, path)
  }

  /// Returns the memory size of this emulator.
  @inlinable
  public func memorySize() -> Int {
    Int(getRAMSize(handle))
  }

  /// Returns a shaped array filled with the contents of this emulator's memory.
  @inlinable
  public func memory() -> ShapedArray<UInt8> {
    let size = Int(getRAMSize(handle))
    let memory = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
    defer { memory.deallocate() }
    getRAM(handle, memory)
    let memoryArray = [UInt8](UnsafeBufferPointer(start: memory, count: size))
    return ShapedArray(shape: [size], scalars: memoryArray)
  }

  /// Saves the state of the emulator.
  @inlinable
  public func saveState() {
    CArcadeLearningEnvironment.saveState(handle)
  }

  /// Loads the state of the emulator.
  @inlinable
  public func loadState() {
    CArcadeLearningEnvironment.loadState(handle)
  }

  /// Gets/sets a string-valued parameter of this emulator.
  ///
  /// - Parameter key: Key corresponding to the parameter.
  @inlinable
  public subscript(_ key: String) -> String {
    get {
      guard let cString = getString(handle, key) else { return "" }
      defer { cString.deallocate() }
      return String(cString: cString)
    }
    set { setString(handle, key, newValue) }
  }

  /// Gets/sets an integer-valued parameter of this emulator.
  ///
  /// - Parameter key: Key corresponding to the parameter.
  @inlinable
  public subscript(_ key: String) -> Int {
    get { Int(getInt(handle, key)) }
    set { setInt(handle, key, Int32(newValue)) }
  }

  /// Gets/sets a boolean-valued parameter of this emulator.
  ///
  /// - Parameter key: Key corresponding to the parameter.
  @inlinable
  public subscript(_ key: String) -> Bool {
    get { getBool(handle, key) }
    set { setBool(handle, key, newValue) }
  }

  /// Gets/sets a float-valued parameter of this emulator.
  ///
  /// - Parameter key: Key corresponding to the parameter.
  @inlinable
  public subscript(_ key: String) -> Float {
    get { getFloat(handle, key) }
    set { setFloat(handle, key, newValue) }
  }
}

extension ArcadeEmulator {
  /// Logging mode for emulators.
  public enum LoggingMode: Int32 {
    case info = 0, warning, error
  }
}

extension ArcadeEmulator {
  /// Emulator state.
  public final class State {
    @usableFromInline internal var handle: UnsafeMutablePointer<ALEState?>?

    /// Creates an emulator state that wraps around the provided native emulator state pointer.
    @inlinable
    internal init(handle: UnsafeMutablePointer<ALEState?>?) {
      self.handle = handle
    }

    /// Creates a new emulator state by decoding the provided sequence of bytes.
    @inlinable
    public init(decoding encoded: [Int8]) {
      encoded.withUnsafeBufferPointer { pointer in
        self.handle = decodeState(pointer.baseAddress!, Int32(encoded.count))
      }
    }

    @inlinable
    deinit {
      if let h = handle { deleteState(h) }
    }

    /// Returns an encoded representation of this state as a byte array.
    @inlinable
    public func encoded() -> [Int8] {
      let size = Int(encodeStateLen(handle))
      let bytes = UnsafeMutablePointer<Int8>.allocate(capacity: size)
      defer { bytes.deallocate() }
      encodeState(handle, bytes, Int32(size))
      return [Int8](UnsafeBufferPointer(start: bytes, count: size))
    }
  }
}

extension ArcadeEmulator {
  /// Actions that an agent can take in this emulator.
  public enum Action: Int32 {
    case noOp = 0, fire, up, right, left, down, upRight, upLeft, downRight, downLeft,
      upFire, rightFire, leftFire, downFire, upRightFire, upLeftFire, downRightFire, downLeftFire
  }
}

extension ArcadeEmulator {
  public enum ScreenFormat {
    /// Raw pixel values from the emulator screen, before any conversion (e.g., to RGB)
    /// takes place. The screen tensor shape is `[height * width]`, where `height` and `width` are
    /// the screen dimensions returned by `Emulator.screenSize()`.
    case raw

    /// Pixel values in RGB format. The screen tensor shape is `[height, width, 3]`, where `height`
    /// and `width` are the screen dimensions returned by `Emulator.screenSize()`.
    case rgb

    /// Pixel values in Grayscale format. The screen tensor shape is `[height, width, 1]`, where
    /// `height` and `width` are the screen dimensions returned by `Emulator.screenSize()`.
    case grayscale

    /// Returns the total number of pixels in the screen.
    @inlinable
    public func size(height: Int, width: Int) -> Int {
      switch self {
      case .raw: return height * width
      case .rgb: return height * width * 3
      case .grayscale: return height * width
      }
    }

    /// Returns the shape of the screen tensor.
    @inlinable
    public func shape(height: Int, width: Int) -> [Int] {
      switch self {
      case .raw: return [height * width]
      case .rgb: return [height, width, 3]
      case .grayscale: return [height, width, 1]
      }
    }
  }
}

extension ArcadeEmulator {
  /// Games currently supported by the arcade emulator.
  public enum Game: String, CaseIterable {
    case adventure = "adventure"
    case airRaid = "air_raid"
    case alien = "alien"
    case amidar = "amidar"
    case assault = "assault"
    case asterix = "asterix"
    case asteroids = "asteroids"
    case atlantis = "atlantis"
    case bankHeist = "bank_heist"
    case battleZone = "battle_zone"
    case beamRider = "beam_rider"
    case berzerk = "berzerk"
    case bowling = "bowling"
    case boxing = "boxing"
    case breakout = "breakout"
    case carnival = "carnival"
    case centipede = "centipede"
    case chopperCommand = "chopper_command"
    case crazyClimber = "crazy_climber"
    case defender = "defender"
    case demonAttack = "demon_attack"
    case doubleDunk = "double_dunk"
    case elevatorAction = "elevator_action"
    case enduro = "enduro"
    case fishingDerby = "fishing_derby"
    case freeway = "freeway"
    case frostbite = "frostbite"
    case gopher = "gopher"
    case gravitar = "gravitar"
    case hero = "hero"
    case iceHockey = "ice_hockey"
    case jamesBond = "jamesbond"
    case journeyEscape = "journey_escape"
    case kaboom = "kaboom"
    case kangaroo = "kangaroo"
    case krull = "krull"
    case kungFuMaster = "kung_fu_master"
    case montezumaRevenge = "montezuma_revenge"
    case msPacman = "ms_pacman"
    case nameThisGame = "name_this_game"
    case phoenix = "phoenix"
    case pitfall = "pitfall"
    case pong = "pong"
    case pooyan = "pooyan"
    case privateEye = "private_eye"
    case qbert = "qbert"
    case riverRaid = "riverraid"
    case roadRunner = "road_runner"
    case robotank = "robotank"
    case seaquest = "seaquest"
    case skiing = "skiing"
    case solaris = "solaris"
    case spaceInvaders = "space_invaders"
    case starGunner = "star_gunner"
    case tennis = "tennis"
    case timePilot = "time_pilot"
    case tutankham = "tutankham"
    case upNDown = "up_n_down"
    case venture = "venture"
    case videoPinball = "video_pinball"
    case wizardOfWor = "wizard_of_wor"
    case yarsRevenge = "yars_revenge"
    case zaxxon = "zaxxon"

    /// Returns the complete URL to this game's ROM.
    ///
    /// - Parameter romPath: Path to the folder in which the game ROMs are stored.
    /// - Returns: Complete URL to this game's ROM.
    /// - Note: If the game ROM cannot be found, an attempt will be made to download it
    ///   automatically. If the download succeeds, then the downloaded ROM file will be placed in
    ///   `gameROMsPath`.
    public func romPath(in gameROMsPath: URL) throws -> URL {
      let fileURL = gameROMsPath.appendingPathComponent("\(rawValue).bin")
      let atariPyGitHub = "https://github.com/openai/atari-py/blob/master/atari_py/atari_roms"
      let gitHubURL = URL(string: "\(atariPyGitHub)/\(rawValue).bin?raw=true")!
      try maybeDownload(from: gitHubURL, to: fileURL)
      return fileURL
    }
  }
}
