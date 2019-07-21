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
import TensorFlow

public final class ArcadeEmulator {
  @usableFromInline internal var handle: UnsafeMutablePointer<ALEInterface?>?

  @inlinable
  public init() {
    self.handle = ALE_new()
  }

  @inlinable
  deinit {
    ALE_del(handle)
  }

  /// State of this emulator.
  /// - Note: This state does *not* include pseudorandomness, making it suitable for planning
  ///   purposes. In contrast, see `Emulator.systemState()`.
  @inlinable
  public var state: State {
    get { State(handle: cloneState(handle)) }
    set { restoreState(handle, newValue.handle) }
  }

  /// State of this emulator that is suitable for serialization.
  /// - Note: This state includes pseudorandomness, making it unsuitable for planning purposes.
  ///   In contrast, see `Emulator.state()`.
  @inlinable
  public var systemState: State {
    get { State(handle: cloneSystemState(handle)) }
    set { restoreSystemState(handle, newValue.handle) }
  }

  @inlinable
  public subscript(_ key: String) -> String {
    get {
      guard let cString = getString(handle, key) else { return "" }
      defer { cString.deallocate() }
      return String(cString: cString)
    }
    set { setString(handle, key, newValue) }
  }

  @inlinable
  public subscript(_ key: String) -> Int {
    get { Int(getInt(handle, key)) }
    set { setInt(handle, key, Int32(newValue)) }
  }

  @inlinable
  public subscript(_ key: String) -> Bool {
    get { getBool(handle, key) }
    set { setBool(handle, key, newValue) }
  }

  @inlinable
  public subscript(_ key: String) -> Float {
    get { getFloat(handle, key) }
    set { setFloat(handle, key, newValue) }
  }

  @inlinable
  public func loadGameROM(from path: String) {
    loadROM(handle, path)
  }

  @inlinable
  public func setMode(_ mode: Int) {
    CArcadeLearningEnvironment.setMode(handle, Int32(mode))
  }

  @inlinable
  public func setDifficulty(_ difficulty: Int) {
    CArcadeLearningEnvironment.setDifficulty(handle, Int32(difficulty))
  }

  @inlinable
  public func resetGame() {
    reset_game(handle)
  }

  @inlinable
  public func takeAction(using action: Int) -> Int {
    Int(act(handle, Int32(action)))
  }

  @inlinable
  public func gameOver() -> Bool {
    game_over(handle)
  }

  @inlinable
  public func lives() -> Int {
    Int(CArcadeLearningEnvironment.lives(handle))
  }

  @inlinable
  public func frameNumber() -> Int {
    Int(getFrameNumber(handle))
  }

  @inlinable
  public func episodeFrameNumber() -> Int {
    Int(getEpisodeFrameNumber(handle))
  }

  @inlinable
  public func availableModes() -> [Int] {
    let count = Int(getAvailableModesSize(handle))
    let modes = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    getAvailableModes(handle, modes)
    return [Int32](UnsafeBufferPointer(start: modes, count: count)).map(Int.init)
  }

  @inlinable
  public func availableDifficulties() -> [Int] {
    let count = Int(getAvailableDifficultiesSize(handle))
    let difficulties = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    getAvailableDifficulties(handle, difficulties)
    return [Int32](UnsafeBufferPointer(start: difficulties, count: count)).map(Int.init)
  }

  @inlinable
  public func legalActions() -> [Int] {
    let count = Int(getLegalActionSize(handle))
    let actions = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    getLegalActionSet(handle, actions)
    return [Int32](UnsafeBufferPointer(start: actions, count: count)).map(Int.init)
  }

  @inlinable
  public func minimalActions() -> [Int] {
    let count = Int(getMinimalActionSize(handle))
    let actions = UnsafeMutablePointer<Int32>.allocate(capacity: count)
    getMinimalActionSet(handle, actions)
    return [Int32](UnsafeBufferPointer(start: actions, count: count)).map(Int.init)
  }

  /// Returns the screen size of this emulator.
  @inlinable
  public func screenSize() -> (height: Int, width: Int) {
    (height: Int(getScreenHeight(handle)), width: Int(getScreenWidth(handle)))
  }

  /// Returns a tensor filled with the pixel data from this emulator's screen.
  ///
  /// - Parameter format: Screen format to use (e.g., RGB pixel values).
  /// - Returns: Tensor with shape determined by the requested `format`.
  @inlinable
  public func screen(format: ScreenFormat = .rgb) -> Tensor<UInt8> {
    let (height, width) = screenSize()
    let size = format.size(height: height, width: width)
    let screen = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
    switch format {
    case .raw: getScreen(handle, screen)
    case .rgb: getScreenRGB(handle, screen)
    case .grayscale: getScreenGrayscale(handle, screen)
    }
    let screenArray = [UInt8](UnsafeBufferPointer(start: screen, count: size))
    return Tensor(shape: format.shape(height: height, width: width), scalars: screenArray)
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

  /// Returns a tensor filled with the contents of this emulator's memory.
  @inlinable
  public func memory() -> Tensor<UInt8> {
    let size = Int(getRAMSize(handle))
    let actions = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
    getRAM(handle, actions)
    let memoryArray = [UInt8](UnsafeBufferPointer(start: actions, count: size))
    return Tensor(shape: [size], scalars: memoryArray)
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

  /// Sets the logging mode for this emulator.
  @inlinable
  public func setLoggingMode(_ mode: LoggingMode) {
    setLoggerMode(mode.rawValue)
  }
}

extension ArcadeEmulator {
  /// Emulator state.
  public final class State {
    @usableFromInline internal var handle: UnsafeMutablePointer<ALEState?>?

    @inlinable
    internal init(handle: UnsafeMutablePointer<ALEState?>?) {
      self.handle = handle
    }

    @inlinable
    public init(encoded: [Int8]) {
      encoded.withUnsafeBufferPointer { pointer in
        self.handle = decodeState(pointer.baseAddress!, Int32(encoded.count))
      }
    }

    @inlinable
    deinit {
      deleteState(handle)
    }

    /// Returns an encoded representation of this state as a byte array.
    @inlinable
    public func encode() -> [Int8] {
      let size = Int(encodeStateLen(handle))
      let bytes = UnsafeMutablePointer<Int8>.allocate(capacity: size)
      encodeState(handle, bytes, Int32(size))
      return [Int8](UnsafeBufferPointer(start: bytes, count: size))
    }
  }

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

    @inlinable
    public func size(height: Int, width: Int) -> Int {
      switch self {
      case .raw: return height * width
      case .rgb: return height * width * 3
      case .grayscale: return height * width
      }
    }

    @inlinable
    public func shape(height: Int, width: Int) -> TensorShape {
      switch self {
      case .raw: return TensorShape(height, width)
      case .rgb: return TensorShape(height, width, 3)
      case .grayscale: return TensorShape(height, width, 1)
      }
    }
  }

  /// Logging mode for emulators.
  public enum LoggingMode: Int32 {
    case info = 0, warning, error
  }
}
