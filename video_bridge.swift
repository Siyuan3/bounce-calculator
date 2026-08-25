import AVFoundation
import Foundation
import ImageIO
import Vision

enum BridgeError: Error, CustomStringConvertible {
    case usage(String)
    case failure(String)

    var description: String {
        switch self {
        case .usage(let message), .failure(let message): return message
        }
    }
}

func number(_ value: String, named name: String) throws -> Double {
    guard let result = Double(value) else {
        throw BridgeError.usage("\(name) 不是有效数字：\(value)")
    }
    return result
}

func videoTrack(_ asset: AVURLAsset) throws -> AVAssetTrack {
    guard let track = asset.tracks(withMediaType: .video).first else {
        throw BridgeError.failure("视频中没有可读取的视频轨道")
    }
    return track
}

func writeFrame(videoPath: String, seconds: Double, outputPath: String) throws {
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    let time = CMTime(seconds: seconds, preferredTimescale: 600)
    var actualTime = CMTime.invalid
    let image = try generator.copyCGImage(at: time, actualTime: &actualTime)

    let outputURL = URL(fileURLWithPath: outputPath) as CFURL
    guard let destination = CGImageDestinationCreateWithURL(
        outputURL, "public.png" as CFString, 1, nil
    ) else {
        throw BridgeError.failure("无法创建标注帧：\(outputPath)")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw BridgeError.failure("无法写入标注帧：\(outputPath)")
    }
    print(String(format: "%.9f", CMTimeGetSeconds(actualTime)))
}

func writeFrames(
    videoPath: String,
    timestampsPath: String,
    outputDirectory: String
) throws {
    let text = try String(contentsOfFile: timestampsPath, encoding: .utf8)
    let timestamps = try text
        .split(whereSeparator: \.isWhitespace)
        .map { value -> Double in
            guard let timestamp = Double(value) else {
                throw BridgeError.usage("无效的帧时间戳：\(value)")
            }
            return timestamp
        }
    guard !timestamps.isEmpty else {
        throw BridgeError.usage("帧时间戳列表为空")
    }
    for (previous, current) in zip(timestamps, timestamps.dropFirst()) {
        guard current > previous else {
            throw BridgeError.usage("请求帧时间戳必须严格递增且不能重复")
        }
    }

    let outputURL = URL(fileURLWithPath: outputDirectory, isDirectory: true)
    try FileManager.default.createDirectory(
        at: outputURL,
        withIntermediateDirectories: true
    )
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero

    var manifest = ["selection_index,requested_timestamp,actual_timestamp,filename"]
    for (index, timestamp) in timestamps.enumerated() {
        let requestedTime = CMTime(seconds: timestamp, preferredTimescale: 60000)
        var actualTime = CMTime.invalid
        let image = try generator.copyCGImage(
            at: requestedTime,
            actualTime: &actualTime
        )
        let filename = String(format: "frame_%04d.jpg", index)
        let frameURL = outputURL.appendingPathComponent(filename) as CFURL
        guard let destination = CGImageDestinationCreateWithURL(
            frameURL,
            "public.jpeg" as CFString,
            1,
            nil
        ) else {
            throw BridgeError.failure("无法创建事件标注帧：\(filename)")
        }
        let options = [
            kCGImageDestinationLossyCompressionQuality: 0.82
        ] as CFDictionary
        CGImageDestinationAddImage(destination, image, options)
        guard CGImageDestinationFinalize(destination) else {
            throw BridgeError.failure("无法写入事件标注帧：\(filename)")
        }
        manifest.append(
            String(
                format: "%d,%.9f,%.9f,%@",
                index,
                timestamp,
                CMTimeGetSeconds(actualTime),
                filename
            )
        )
    }
    try manifest.joined(separator: "\n").appending("\n").write(
        to: outputURL.appendingPathComponent("manifest.csv"),
        atomically: true,
        encoding: .utf8
    )
}

func writeVideoInfo(videoPath: String) throws {
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let track = try videoTrack(asset)
    let duration = CMTimeGetSeconds(asset.duration)
    let size = track.naturalSize.applying(track.preferredTransform)
    print(
        String(
            format: "{\"path\":\"%@\",\"duration_s\":%.9f,\"nominal_fps\":%.6f,\"width\":%.0f,\"height\":%.0f,\"natural_width\":%.0f,\"natural_height\":%.0f,\"transform\":[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]}",
            videoPath,
            duration,
            track.nominalFrameRate,
            abs(size.width),
            abs(size.height),
            track.naturalSize.width,
            track.naturalSize.height,
            track.preferredTransform.a,
            track.preferredTransform.b,
            track.preferredTransform.c,
            track.preferredTransform.d,
            track.preferredTransform.tx,
            track.preferredTransform.ty
        )
    )
}

func writeTimestamps(
    videoPath: String,
    startSeconds: Double,
    endSeconds: Double,
    outputPath: String
) throws {
    guard endSeconds > startSeconds else {
        throw BridgeError.usage("end 必须大于 start")
    }
    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let track = try videoTrack(asset)
    let reader = try AVAssetReader(asset: asset)
    reader.timeRange = CMTimeRange(
        start: CMTime(seconds: startSeconds, preferredTimescale: 60000),
        end: CMTime(seconds: endSeconds, preferredTimescale: 60000)
    )
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw BridgeError.failure("无法创建视频时间戳读取器")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw BridgeError.failure("无法读取视频时间戳")
    }
    var timestamps: [Double] = []
    while let sample = output.copyNextSampleBuffer() {
        let timestamp = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
        // Some iPhone slow-motion assets ignore AVAssetReader.timeRange around
        // edit-list boundaries and return samples from the start of the track.
        // Explicitly filter presentation timestamps so callers never receive
        // frames outside the requested analysis interval.
        let tolerance = 1.0 / 60000.0
        if timestamp.isFinite
            && timestamp >= startSeconds - tolerance
            && timestamp <= endSeconds + tolerance {
            timestamps.append(timestamp)
        }
    }
    guard reader.status == .completed else {
        throw BridgeError.failure("视频时间戳读取未完成")
    }
    let sorted = Array(Set(timestamps)).sorted()
    try sorted.map { String(format: "%.9f", $0) }
        .joined(separator: "\n").appending("\n").write(
        to: URL(fileURLWithPath: outputPath), atomically: true, encoding: .utf8
    )
}

func visionOrientation(for transform: CGAffineTransform) -> CGImagePropertyOrientation {
    let epsilon: CGFloat = 0.01
    func close(_ lhs: CGFloat, _ rhs: CGFloat) -> Bool {
        return abs(lhs - rhs) < epsilon
    }
    if close(transform.a, 0) && close(transform.b, 1)
        && close(transform.c, -1) && close(transform.d, 0) {
        return .right
    }
    if close(transform.a, 0) && close(transform.b, -1)
        && close(transform.c, 1) && close(transform.d, 0) {
        return .left
    }
    if close(transform.a, -1) && close(transform.d, -1) {
        return .down
    }
    return .up
}

func trackBall(
    videoPath: String,
    startSeconds: Double,
    endSeconds: Double,
    roiPixels: CGRect,
    outputPath: String
) throws {
    guard endSeconds > startSeconds else {
        throw BridgeError.usage("end 必须大于 start")
    }

    let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
    let track = try videoTrack(asset)
    let reader = try AVAssetReader(asset: asset)
    reader.timeRange = CMTimeRange(
        start: CMTime(seconds: startSeconds, preferredTimescale: 600),
        end: CMTime(seconds: endSeconds, preferredTimescale: 600)
    )
    let settings: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
    ]
    let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
    output.alwaysCopiesSampleData = false
    guard reader.canAdd(output) else {
        throw BridgeError.failure("无法创建视频帧读取器")
    }
    reader.add(output)
    guard reader.startReading() else {
        throw BridgeError.failure("无法开始读取视频：\(reader.error?.localizedDescription ?? "未知错误")")
    }

    var timestamps: [Double] = []
    while let sample = output.copyNextSampleBuffer() {
        autoreleasepool {
            timestamps.append(
                CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
            )
        }
    }

    guard reader.status == .completed else {
        throw BridgeError.failure("视频读取未完成：\(reader.error?.localizedDescription ?? "未知错误")")
    }
    guard timestamps.count >= 20 else {
        throw BridgeError.failure("视频时间戳数量不足：\(timestamps.count)")
    }

    // AVAssetReader exposes encoded landscape buffers for portrait MOV files,
    // while the annotation UI displays the preferred transform.  Generate the
    // exact presentation frames with that transform applied so Vision and the
    // human-drawn ROI share one coordinate system.
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    let handler = VNSequenceRequestHandler()
    var request: VNTrackObjectRequest?
    var rows = ["timestamp,x_pixel,y_pixel,width_pixel,height_pixel,confidence"]
    var frameCount = 0

    for timestamp in timestamps {
        autoreleasepool {
            do {
                let requested = CMTime(seconds: timestamp, preferredTimescale: 60000)
                var actual = CMTime.invalid
                let image = try generator.copyCGImage(at: requested, actualTime: &actual)
                let width = CGFloat(image.width)
                let height = CGFloat(image.height)
                if request == nil {
                    let normalized = CGRect(
                        x: roiPixels.minX / width,
                        y: 1.0 - roiPixels.maxY / height,
                        width: roiPixels.width / width,
                        height: roiPixels.height / height
                    )
                    let observation = VNDetectedObjectObservation(boundingBox: normalized)
                    let newRequest = VNTrackObjectRequest(detectedObjectObservation: observation)
                    newRequest.trackingLevel = .accurate
                    request = newRequest
                    rows.append(String(format: "%.9f,%.4f,%.4f,%.4f,%.4f,1.000000",
                                       timestamp, roiPixels.midX, roiPixels.midY,
                                       roiPixels.width, roiPixels.height))
                    frameCount += 1
                    return
                }

                guard let activeRequest = request else { return }
                try handler.perform([activeRequest], on: image, orientation: .up)
                guard let observation = activeRequest.results?.first as? VNDetectedObjectObservation else {
                    return
                }
                let box = observation.boundingBox
                let x = box.midX * width
                let y = (1.0 - box.midY) * height
                rows.append(String(format: "%.9f,%.4f,%.4f,%.4f,%.4f,%.6f",
                                   timestamp, x, y, box.width * width,
                                   box.height * height, observation.confidence))
                activeRequest.inputObservation = observation
                frameCount += 1
            } catch {
                fputs("跟踪帧失败 \(timestamp)：\(error)\n", stderr)
            }
        }
    }
    guard frameCount >= 20 else {
        throw BridgeError.failure("自动跟踪得到的帧数不足：\(frameCount)")
    }
    try rows.joined(separator: "\n").appending("\n").write(
        to: URL(fileURLWithPath: outputPath), atomically: true, encoding: .utf8
    )
}

do {
    let arguments = CommandLine.arguments
    guard arguments.count >= 2 else {
        throw BridgeError.usage("用法：video_bridge.swift frame|track ...")
    }

    switch arguments[1] {
    case "info":
        guard arguments.count >= 3 else {
            throw BridgeError.usage("info 用法：info VIDEO [VIDEO ...]")
        }
        for videoPath in arguments.dropFirst(2) {
            try writeVideoInfo(videoPath: videoPath)
        }
    case "frame":
        guard arguments.count == 5 else {
            throw BridgeError.usage("frame 用法：frame VIDEO SECONDS OUTPUT.png")
        }
        try writeFrame(
            videoPath: arguments[2],
            seconds: try number(arguments[3], named: "seconds"),
            outputPath: arguments[4]
        )
    case "timestamps":
        guard arguments.count == 6 else {
            throw BridgeError.usage(
                "timestamps 用法：timestamps VIDEO START END OUTPUT.txt"
            )
        }
        try writeTimestamps(
            videoPath: arguments[2],
            startSeconds: try number(arguments[3], named: "start"),
            endSeconds: try number(arguments[4], named: "end"),
            outputPath: arguments[5]
        )
    case "track":
        guard arguments.count == 10 else {
            throw BridgeError.usage(
                "track 用法：track VIDEO START END X Y WIDTH HEIGHT OUTPUT.csv"
            )
        }
        try trackBall(
            videoPath: arguments[2],
            startSeconds: try number(arguments[3], named: "start"),
            endSeconds: try number(arguments[4], named: "end"),
            roiPixels: CGRect(
                x: try number(arguments[5], named: "x"),
                y: try number(arguments[6], named: "y"),
                width: try number(arguments[7], named: "width"),
                height: try number(arguments[8], named: "height")
            ),
            outputPath: arguments[9]
        )
    case "frames":
        guard arguments.count == 5 else {
            throw BridgeError.usage(
                "frames 用法：frames VIDEO TIMESTAMPS.txt OUTPUT_DIRECTORY"
            )
        }
        try writeFrames(
            videoPath: arguments[2],
            timestampsPath: arguments[3],
            outputDirectory: arguments[4]
        )
    default:
        throw BridgeError.usage("未知操作：\(arguments[1])")
    }
} catch {
    fputs("错误：\(error)\n", stderr)
    exit(1)
}
