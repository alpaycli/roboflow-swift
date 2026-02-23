//
//  RFObjectDetectionModel.swift
//  Roboflow
//
//  Created by Nicholas Arner on 4/12/22.
//

#if canImport(UIKit)
import UIKit
#endif
import Foundation
import CoreML
import Vision
//Creates an instance of an ML model that's hosted on Roboflow
public class RFObjectDetectionModel: RFModel {

    public override init() {
        super.init()
    }
    
    //Stores the retreived ML model
    var thresholdProvider = ThresholdProvider()
    
    //Configure the parameters for the model
    public override func configure(threshold: Double = 0.5, overlap: Double = 0.5, maxObjects: Float = 20, processingMode: ProcessingMode = .balanced, maxNumberPoints: Int = 500) {
        super.configure(threshold: threshold, overlap: overlap, maxObjects: maxObjects, processingMode: processingMode, maxNumberPoints: maxNumberPoints)
        thresholdProvider.values = ["iouThreshold": MLFeatureValue(double: self.overlap),
                                    "confidenceThreshold": MLFeatureValue(double: self.threshold)]
        if visionModel != nil {
            if #available(macOS 10.15, *) {
                visionModel.featureProvider = thresholdProvider
            } else {
                // Fallback on earlier versions
            }
        }
    }
    
    //Load the retrieved CoreML model into an already created RFObjectDetectionModel instance
    override func loadMLModel(modelPath: URL, colors: [String: String], classes: [String], environment: [String: Any]) -> Error? {
        let _ = super.loadMLModel(modelPath: modelPath, colors: colors, classes: classes, environment: environment)
        do {
            if #available(macOS 10.14, *) {
                let config = MLModelConfiguration()
                if #available(macOS 10.15, *) {
                    mlModel = try yolov5s(contentsOf: modelPath, configuration: config).model
                } else {
                    // Fallback on earlier versions
                    return UnsupportedOSError()
                }
                visionModel = try VNCoreMLModel(for: mlModel)
                if #available(macOS 10.15, *) {
                    visionModel.featureProvider = thresholdProvider
                } else {
                    // Fallback on earlier versions
                    return UnsupportedOSError()
                }
                let request = VNCoreMLRequest(model: visionModel)
                request.imageCropAndScaleOption = .scaleFill
                coreMLRequest = request
            } else {
                // Fallback on earlier versions
                return UnsupportedOSError()
            }
            
        } catch {
            return error
        }
        return nil
    }
    
    //Run image through model and return Detections
    @available(*, renamed: "detect(image:)")
    public override func detect(pixelBuffer buffer: CVPixelBuffer, completion: @escaping (([RFPrediction]?, Error?) -> Void)) {
        guard let coreMLRequest = self.coreMLRequest else {
            completion(nil, "Model initialization failed.")
            return
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)

        do {
            try handler.perform([coreMLRequest])
            
            guard let detectResults = coreMLRequest.results as? [VNDetectedObjectObservation] else { return }
            
            var detections:[RFObjectDetectionPrediction] = []
            for detectResult in detectResults {
                let flippedBox = CGRect(x: detectResult.boundingBox.minX, y: 1 - detectResult.boundingBox.maxY, width: detectResult.boundingBox.width, height: detectResult.boundingBox.height)
                
                let box = VNImageRectForNormalizedRect(flippedBox, Int(buffer.width()), Int(buffer.height()))
                let confidence = detectResult.confidence
                var label:String = ""
                if #available(macOS 10.14, *) {
                    if let recognizedResult = detectResult as? VNRecognizedObjectObservation, let classLabel = recognizedResult.labels.first?.identifier {
                        // class labels may be stripped from the model when weights trained externally and uploaded.
                        // if our class it is an integer and it is not a defined class, look it up in our class map.
                        if let intValue = Int(classLabel), !classes.contains(classLabel), intValue < classes.count {
                            label = classes[intValue]
                        } else {
                            label = classLabel
                        }
                    }
                } else {
                    // Fallback on earlier versions
                    completion(nil, UnsupportedOSError())
                    return
                }
                let detection = RFObjectDetectionPrediction(x: Float((box.maxX+box.minX)/2.0), y: Float((box.maxY+box.minY)/2.0), width: Float((box.maxX-box.minX)), height: Float((box.maxY-box.minY)), className: label, confidence: confidence, color: hexStringToCGColor(hex: colors[label] ?? "#ff0000"), box: box)
                detections.append(detection)
            }
            completion(detections, nil)
        } catch let error {
            completion(nil, error)
        }
    }
   
   #if canImport(UIKit)
   @available(*, renamed: "detect(image:)")
   public override func detect(
       pixelBuffer buffer: CVPixelBuffer,
       options: RFDetectionOptions = RFDetectionOptions(),
       completion: @escaping (([RFPrediction]?, Error?) -> Void)
   ) {
       guard let coreMLRequest = self.coreMLRequest else {
           completion(nil, "Model initialization failed.")
           return
       }

       // Pass orientation so Vision rotates coordinates into portrait/UI space
       let handler = VNImageRequestHandler(
           cvPixelBuffer: buffer,
           orientation: options.orientation
       )

       do {
           try handler.perform([coreMLRequest])

           guard let detectResults = coreMLRequest.results as? [VNDetectedObjectObservation] else { return }

           var detections: [RFObjectDetectionPrediction] = []

           for detectResult in detectResults {
               // Skip low-confidence results early
               guard detectResult.confidence >= options.confidenceThreshold else { continue }

               var boundingBox = detectResult.boundingBox

               // Mirror X axis for front camera (Vision doesn't do this automatically)
               if options.cameraPosition == .front {
                   boundingBox.origin.x = 1 - boundingBox.maxX
               }

               // Apply optional padding
               if options.boundingBoxPadding != 1.0 {
                   let dw = boundingBox.width * CGFloat(options.boundingBoxPadding - 1.0) / 2
                   let dh = boundingBox.height * CGFloat(options.boundingBoxPadding - 1.0) / 2
                   boundingBox = boundingBox.insetBy(dx: -dw, dy: -dh)
               }

               // Flip Y: Vision is bottom-left origin, UIKit is top-left
//               let flippedBox = CGRect(
//                   x: boundingBox.minX,
//                   y: 1 - boundingBox.maxY,
//                   width: boundingBox.width,
//                   height: boundingBox.height
//               )

               let confidence = detectResult.confidence

               var label: String = ""
               if #available(macOS 10.14, *) {
                   if let recognizedResult = detectResult as? VNRecognizedObjectObservation,
                      let classLabel = recognizedResult.labels.first?.identifier {
                       if let intValue = Int(classLabel), !classes.contains(classLabel), intValue < classes.count {
                           label = classes[intValue]
                       } else {
                           label = classLabel
                       }
                   }
               } else {
                   completion(nil, UnsupportedOSError())
                   return
               }


              let detection = RFObjectDetectionPrediction(
                  x: Float(boundingBox.midX),
                  y: Float(boundingBox.midY),
                  width: Float(boundingBox.width),
                  height: Float(boundingBox.height),
                  className: label,
                  confidence: confidence,
                  color: hexStringToCGColor(hex: colors[label] ?? "#ff0000"),
                  box: boundingBox
              )
              detections.append(detection)
           }

           completion(detections, nil)

       } catch {
           completion(nil, error)
       }
   }
   
   func exifOrientationForDeviceOrientation(_ deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
       
       switch deviceOrientation {
       case .portraitUpsideDown:
           return .rightMirrored
           
       case .landscapeLeft:
           return .downMirrored
           
       case .landscapeRight:
           return .upMirrored
           
       default:
           return .leftMirrored
       }
   }
   
   func exifOrientationForCurrentDeviceOrientation() -> CGImagePropertyOrientation {
       return exifOrientationForDeviceOrientation(UIDevice.current.orientation)
   }
   #endif
}

import AVFoundation

public struct RFDetectionOptions {
    /// Which camera the buffer came from — affects horizontal mirroring
    public var cameraPosition: AVCaptureDevice.Position
    /// Physical device orientation when the frame was captured
    public var orientation: CGImagePropertyOrientation
    /// Discard detections below this threshold (0.0 – 1.0)
    public var confidenceThreshold: Float
    /// Scale bounding boxes outward by this factor (e.g. 1.1 = 10% padding)
    public var boundingBoxPadding: Float

    public init(
        cameraPosition: AVCaptureDevice.Position = .back,
        orientation: CGImagePropertyOrientation = .right,
        confidenceThreshold: Float = 0.5,
        boundingBoxPadding: Float = 1.0
    ) {
        self.cameraPosition = cameraPosition
        self.orientation = orientation
        self.confidenceThreshold = confidenceThreshold
        self.boundingBoxPadding = boundingBoxPadding
    }
}
