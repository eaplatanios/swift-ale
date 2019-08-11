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
import TensorFlow

#if os(Linux)
import FoundationNetworking
#endif

/// Creates an orthogonal matrix or tensor. 
///
/// If the shape of the tensor to initialize is two-dimensional, it is initialized with an 
/// orthogonal matrix obtained from the QR decomposition of a matrix of random numbers drawn 
/// from a normal distribution. If the matrix has fewer rows than columns then the output will 
/// have orthogonal rows. Otherwise, the output will have orthogonal columns.
/// 
/// If the shape of the tensor to initialize is more than two-dimensional, a matrix of shape 
/// `[shape[0] * ... * shape[rank - 2], shape[rank - 1]]` is initialized.  The matrix is 
/// subsequently reshaped to give a tensor of the desired shape.
///
/// - Parameters:
///   - shape: The shape of the tensor.
///   - gain: A multiplicative factor to apply to the orthogonal tensor.
///   - seed: A tuple of two integers to seed the random number generator.
public func orthogonal<Scalar: TensorFlowFloatingPoint>(
    gain: Tensor<Scalar> = Tensor<Scalar>(1),
    seed: TensorFlowSeed = Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
    { Tensor<Scalar>(orthogonal: $0, gain: gain, seed: seed) }
}

/// Downloads the file at `url` to `path`, if `path` does not exist.
///
/// - Parameters:
///     - from: URL to download data from.
///     - to: Destination file path.
///
/// - Returns: Boolean value indicating whether a download was
///     performed (as opposed to not needed).
internal func maybeDownload(from url: URL, to destination: URL) throws {
  if !FileManager.default.fileExists(atPath: destination.path) {
    // Create any potentially missing directories.
    try FileManager.default.createDirectory(
      atPath: destination.deletingLastPathComponent().path,
      withIntermediateDirectories: true)

    // Create the URL session that will be used to download the dataset.
    let semaphore = DispatchSemaphore(value: 0)
    let delegate = DataDownloadDelegate(destinationFileUrl: destination, semaphore: semaphore)
    let session = URLSession(configuration: .default, delegate: delegate, delegateQueue: nil)

    // Download the data to a temporary file and then copy that file to
    // the destination path.
    print("Downloading \(url).")
    let task = session.downloadTask(with: url)
    task.resume()

    // Wait for the download to finish.
    semaphore.wait()
  }
}

internal class DataDownloadDelegate: NSObject, URLSessionDownloadDelegate {
  let destinationFileUrl: URL
  let semaphore: DispatchSemaphore
  let numBytesFrequency: Int64

  internal var logCount: Int64 = 0

  init(
    destinationFileUrl: URL,
    semaphore: DispatchSemaphore,
    numBytesFrequency: Int64 = 1024 * 1024
  ) {
    self.destinationFileUrl = destinationFileUrl
    self.semaphore = semaphore
    self.numBytesFrequency = numBytesFrequency
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) -> Void {
    if (totalBytesWritten / numBytesFrequency > logCount) {
      let mBytesWritten = String(format: "%.2f", Float(totalBytesWritten) / (1024 * 1024))
      if totalBytesExpectedToWrite > 0 {
        let mBytesExpectedToWrite = String(
          format: "%.2f", Float(totalBytesExpectedToWrite) / (1024 * 1024))
        print("Downloaded \(mBytesWritten) MBs out of \(mBytesExpectedToWrite).")
      } else {
        print("Downloaded \(mBytesWritten) MBs.")
      }
      logCount += 1
    }
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) -> Void {
    logCount = 0
    do {
      try FileManager.default.moveItem(at: location, to: destinationFileUrl)
    } catch (let writeError) {
      print("Error writing file \(location.path) : \(writeError)")
    }
    print("The file was downloaded successfully to \(location.path).")
    semaphore.signal()
  }
}

@usableFromInline
internal enum ImageResizeMethod {
  case nearestNeighbor
  case bilinear
  case bicubic
  case area

  @inlinable
  internal func resize(
    images: Tensor<Float>,
    size: Tensor<Int32>,
    alignCorners: Bool = false
  ) -> Tensor<Float> {
    switch self {
    case .nearestNeighbor:
      return Raw.resizeNearestNeighbor(images: images, size: size, alignCorners: alignCorners)
    case .bilinear:
      return Raw.resizeBilinear(images: images, size: size, alignCorners: alignCorners)
    case .bicubic:
      return Raw.resizeBicubic(images: images, size: size, alignCorners: alignCorners)
    case .area:
      return Raw.resizeArea(images: images, size: size, alignCorners: alignCorners)
    }
  }
}

@inlinable
internal func resize<Scalar: TensorFlowNumeric>(
  images: Tensor<Scalar>,
  to size: Tensor<Int32>,
  method: ImageResizeMethod = .bilinear,
  alignCorners: Bool = false,
  preserveAspectRatio: Bool = false
) -> Tensor<Scalar> {
  precondition(images.rank == 3 || images.rank == 4, "'images' must be of rank 3 or 4.")
  precondition(size.rank == 1 && size.shape[0] == 2, "'size' must be a vector with 2 elements.")
  let batched = images.rank == 4
  let images = batched ? images : images.expandingShape(at: 0)
  let height = images.shape[1]
  let width = images.shape[2]
  var newHeight = size[0].scalarized()
  var newWidth = size[1].scalarized()

  // Compute appropriate new size based on whether `preserveAspectRatio` is `true`.
  if preserveAspectRatio {
    let heightScaleFactor = Float(newHeight) / Float(height)
    let widthScaleFactor = Float(newWidth) / Float(width)
    let scaleFactor = min(heightScaleFactor, widthScaleFactor)
    newHeight = Int32(scaleFactor * Float(height))
    newWidth = Int32(scaleFactor * Float(width))
  }

  // Check if the resize is necessary.
  if height == newHeight && width == newWidth {
    return batched ? images : images.squeezingShape(at: 0)
  }

  let resizedImages = Tensor<Scalar>(method.resize(
    images: Tensor<Float>(images),
    size: preserveAspectRatio ? Tensor<Int32>([Int32(newHeight), Int32(newWidth)]) : size,
    alignCorners: alignCorners))
  
  return batched ? resizedImages : resizedImages.squeezingShape(at: 0)
}
