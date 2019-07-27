// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

let package = Package(
  name: "ArcadeLearningEnvironment",
  platforms: [.macOS(.v10_12)],
  products: [
    .library(
      name: "ArcadeLearningEnvironment",
      targets: ["ArcadeLearningEnvironment"]),
    .executable(
      name: "ArcadeLearningEnvironmentExperiments",
      targets: ["ArcadeLearningEnvironmentExperiments"]),
  ],
  dependencies: [
    .package(url: "https://github.com/eaplatanios/swift-rl.git", .branch("master")),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
  ],
  targets: [
    .systemLibrary(
      name: "CArcadeLearningEnvironment",
      path: "Sources/CArcadeLearningEnvironment",
      pkgConfig: "ale"),
    .target(
      name: "ArcadeLearningEnvironment",
      dependencies: ["CArcadeLearningEnvironment", "ReinforcementLearning"],
      path: "Sources/ArcadeLearningEnvironment"),
    .target(
      name: "ArcadeLearningEnvironmentExperiments",
      dependencies: ["ArcadeLearningEnvironment", "Logging"],
      path: "Sources/ArcadeLearningEnvironmentExperiments"),
  ]
)
