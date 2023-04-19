import sys
import cv2
import depthai as dai
import numpy as np
import time


nnBlobPath = "best_openvino_2022.1_4shave.blob"

labelMap = ["trunk"]

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutDepth.setStreamName("depth")
nnNetworkOut.setStreamName("nnNetwork")

# Properties
camRgb.setPreviewSize(640, 640)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(2)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
# Align depth map to the perspective of RGB camera, on which inference is done
stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(len(labelMap))
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors([10.0,
                13.0,
                16.0,
                30.0,
                33.0,
                23.0,
                30.0,
                61.0,
                62.0,
                45.0,
                59.0,
                119.0,
                116.0,
                90.0,
                156.0,
                198.0,
                373.0,
                326.0])
spatialDetectionNetwork.setAnchorMasks({"side80": [
                    0,
                    1,
                    2
                ],
                "side40": [
                    3,
                    4,
                    5
                ],
                "side20": [
                    6,
                    7,
                    8
                ]})
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)
spatialDetectionNetwork.outNetwork.link(nnNetworkOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and the spatial detection data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
            inDepth = qDepth.get()

            frame = inRgb.getCvFrame()
            depthFrame = inDepth.getFrame()
            detections = inDet.detections
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()
            inDepth = qDepth.tryGet()

            if inRgb is not None:
                frame = inRgb.getCvFrame()

            if inDet is not None:
                detections = inDet.detections

            if inDepth is not None:
                depthFrame = inDepth.getFrame()

        if frame is not None:
            # Display the resulting frame
            for detection in detections:
                x1, y1, x2, y2 = int(detection.xmin), int(detection.ymin), int(detection.xmax), int(detection.ymax)
                centerX = int((detection.xmin + detection.xmax) / 2)
                centerY = int((detection.ymin + detection.ymax) / 2)

                if detection.label < len(labelMap):
                    label = labelMap[detection.label]
                else:
                    label = "unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centerX, centerY), 2, (0, 0, 255), 2)

                # Get depth value for the center point of the object
                depth_value = depthFrame[centerY, centerX]
                cv2.putText(frame, f"Depth: {depth_value}mm", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Spatial Detection", frame)

        counter += 1
        if (time.monotonic() - startTime) > 1:
            print(f"FPS: {counter / (time.monotonic() - startTime)}")
            counter = 0
            startTime = time.monotonic()

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources and close windows
cv2.destroyAllWindows()




