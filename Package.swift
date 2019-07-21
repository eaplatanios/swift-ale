// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

let package = Package(
  name: "ArcadeLearningEnvironment",
  products: [
    .library(
      name: "ArcadeLearningEnvironment",
      targets: ["ArcadeLearningEnvironment"]),
    .executable(
      name: "ArcadeLearningEnvironmentExperiments",
      targets: ["ArcadeLearningEnvironmentExperiments"]),
  ],
  targets: [
    .systemLibrary(
      name: "CArcadeLearningEnvironment",
      path: "Sources/CArcadeLearningEnvironment",
      pkgConfig: "ale"),
    .target(
      name: "ArcadeLearningEnvironment",
      dependencies: ["CArcadeLearningEnvironment"],
      path: "Sources/ArcadeLearningEnvironment"),
    .target(
      name: "ArcadeLearningEnvironmentExperiments",
      dependencies: ["ArcadeLearningEnvironment"],
      path: "Sources/ArcadeLearningEnvironmentExperiments"),
  ]
)
