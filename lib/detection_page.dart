import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

/// Pantalla de detección en vivo basada en YOLO11n.
///
/// Integra la cámara con CameraX (a través del plugin oficial `camera`) y
/// ejecuta el modelo de detección de objetos directamente en el dispositivo
/// utilizando TensorFlow Lite. Los resultados se dibujan encima de la
/// previsualización y se muestra una estimación heurística de distancia para
/// cada detección.
class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  tfl.Interpreter? _interpreter;
  bool _initializing = true;
  bool _modelLoaded = false;
  String? _error;
  bool _isProcessingFrame = false;
  int _inputWidth = 0;
  int _inputHeight = 0;
  Size? _lastImageSize;
  List<_Detection> _detections = const [];

  static const double _confidenceThreshold = 0.35;
  static const double _iouThreshold = 0.45;
  static final List<String> _labels = List.unmodifiable(_cocoLabels);

  late final bool _supportsYolo = !kIsWeb && defaultTargetPlatform == TargetPlatform.android;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    unawaited(_initialize());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_stopPreview());
    _cameraController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      unawaited(_stopPreview());
    } else if (state == AppLifecycleState.resumed) {
      unawaited(_startPreview());
    }
  }

  Future<void> _initialize() async {
    try {
      if (!_supportsYolo) {
        throw StateError('La detección en vivo solo está disponible en Android.');
      }

      await _loadModel();
      await _initializeCamera();
    } catch (e, stack) {
      debugPrint('No se pudo iniciar la detección: $e\n$stack');
      if (!mounted) return;
      setState(() {
        _error = 'No se pudo iniciar la detección: $e';
        _initializing = false;
      });
    }
  }

  Future<void> _loadModel() async {
    final options = tfl.InterpreterOptions();
    if (_supportsYolo) {
      options.threads = 2;
    }

    final interpreter =
        await tfl.Interpreter.fromAsset('models/yolo11n.tflite', options: options);

    final inputTensor = interpreter.getInputTensor(0);
    final inputShape = inputTensor.shape;
    if (inputShape.length != 4) {
      throw StateError('Forma de entrada inesperada: $inputShape');
    }

    _inputHeight = inputShape[1];
    _inputWidth = inputShape[2];

    setState(() {
      _interpreter = interpreter;
      _modelLoaded = true;
    });
  }

  Future<void> _initializeCamera() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) {
      throw StateError('No se encontró ninguna cámara disponible.');
    }

    final selectedCamera = cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    final controller = CameraController(
      selectedCamera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup:
          kIsWeb ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
    );

    await controller.initialize();
    await controller.setFlashMode(FlashMode.off);

    if (!mounted) {
      await controller.dispose();
      return;
    }

    setState(() {
      _cameraController = controller;
      _initializing = false;
      _error = null;
    });

    await _startPreview();
  }

  Future<void> _startPreview() async {
    final controller = _cameraController;
    final interpreter = _interpreter;
    if (controller == null || interpreter == null) {
      return;
    }

    if (!controller.value.isStreamingImages) {
      try {
        await controller.startImageStream(_processCameraImage);
      } catch (e) {
        debugPrint('No se pudo iniciar el stream de imágenes: $e');
      }
    }
  }

  Future<void> _stopPreview() async {
    final controller = _cameraController;
    if (controller == null) return;

    if (controller.value.isStreamingImages) {
      try {
        await controller.stopImageStream();
      } catch (e) {
        debugPrint('Error al detener el stream de imágenes: $e');
      }
    }
  }

  void _processCameraImage(CameraImage image) {
    if (_isProcessingFrame || !mounted) {
      return;
    }

    final interpreter = _interpreter;
    final controller = _cameraController;
    if (interpreter == null || controller == null) {
      return;
    }

    if (_inputWidth <= 0 || _inputHeight <= 0) {
      return;
    }

    _isProcessingFrame = true;

    Future<void>(() async {
      try {
        image_lib.Image rgbImage = _convertCameraImage(image);
        final rotation = controller.description.sensorOrientation;
        if (rotation != 0) {
          rgbImage = _applyRotation(rgbImage, rotation);
        }

        final preprocess = _preprocess(rgbImage);
        final outputTensor = interpreter.getOutputTensor(0);
        final outputShape = outputTensor.shape;
        final int valuesPerBox = outputShape.last;
        final int numBoxes = outputShape[1];
        final Float32List outputBuffer =
            Float32List(outputShape.reduce((value, element) => value * element));

        interpreter.run(preprocess.input, outputBuffer);

        final detections = _parseDetections(
          outputBuffer,
          numBoxes,
          valuesPerBox,
          preprocess.scale,
          preprocess.padX,
          preprocess.padY,
          rgbImage.width.toDouble(),
          rgbImage.height.toDouble(),
        );

        if (!mounted) return;
        setState(() {
          _detections = detections;
          _lastImageSize = Size(
            rgbImage.width.toDouble(),
            rgbImage.height.toDouble(),
          );
        });
      } catch (e, stack) {
        debugPrint('Error procesando frame: $e\n$stack');
      } finally {
        _isProcessingFrame = false;
      }
    });
  }

  image_lib.Image _convertCameraImage(CameraImage image) {
    if (image.format.group != ImageFormatGroup.yuv420) {
      throw UnsupportedError('Formato no soportado: ${image.format.group}');
    }

    final int width = image.width;
    final int height = image.height;
    final planeY = image.planes[0];
    final planeU = image.planes[1];
    final planeV = image.planes[2];
    final int uvRowStride = planeU.bytesPerRow;
    final int uvPixelStride = planeU.bytesPerPixel ?? 1;

    final bytes = Uint8List(width * height * 3);
    int pixelIndex = 0;

    for (int y = 0; y < height; y++) {
      final int uvRow = uvRowStride * (y >> 1);
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvRow + (x >> 1) * uvPixelStride;
        final int yIndex = y * planeY.bytesPerRow + x;

        final int yValue = planeY.bytes[yIndex];
        final int uValue = planeU.bytes[uvIndex];
        final int vValue = planeV.bytes[uvIndex];

        final double yf = yValue.toDouble();
        final double uf = uValue.toDouble() - 128.0;
        final double vf = vValue.toDouble() - 128.0;

        int r = (yf + 1.370705 * vf).round();
        int g = (yf - 0.337633 * uf - 0.698001 * vf).round();
        int b = (yf + 1.732446 * uf).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        bytes[pixelIndex++] = r;
        bytes[pixelIndex++] = g;
        bytes[pixelIndex++] = b;
      }
    }

    return image_lib.Image.fromBytes(
      width: width,
      height: height,
      bytes: bytes.buffer,
      numChannels: 3,
    );
  }

  image_lib.Image _applyRotation(image_lib.Image source, int rotationDegrees) {
    switch (rotationDegrees % 360) {
      case 90:
        return image_lib.copyRotate(source, angle: 90);
      case 180:
        return image_lib.copyRotate(source, angle: 180);
      case 270:
        return image_lib.copyRotate(source, angle: 270);
      default:
        return source;
    }
  }

  _PreprocessResult _preprocess(image_lib.Image image) {
    final int targetWidth = _inputWidth;
    final int targetHeight = _inputHeight;

    final double scale = math.min(
      targetWidth / image.width,
      targetHeight / image.height,
    );

    final int resizedWidth = (image.width * scale).round();
    final int resizedHeight = (image.height * scale).round();
    final double padX = (targetWidth - resizedWidth) / 2;
    final double padY = (targetHeight - resizedHeight) / 2;

    final image_lib.Image resized = image_lib.copyResize(
      image,
      width: resizedWidth,
      height: resizedHeight,
      interpolation: image_lib.Interpolation.linear,
    );

    final image_lib.Image letterboxed = image_lib.Image(
      width: targetWidth,
      height: targetHeight,
      numChannels: 3,
    );

    image_lib.fill(letterboxed, image_lib.ColorInt8.rgb(0, 0, 0));
    image_lib.copyInto(
      letterboxed,
      resized,
      dstX: padX.round(),
      dstY: padY.round(),
    );

    final Uint8List pixels = letterboxed.getBytes(order: image_lib.ChannelOrder.rgb);
    final Float32List input = Float32List(pixels.length);
    for (int i = 0; i < pixels.length; i++) {
      input[i] = pixels[i] / 255.0;
    }

    return _PreprocessResult(
      input: input,
      scale: scale,
      padX: padX,
      padY: padY,
    );
  }

  List<_Detection> _parseDetections(
    Float32List output,
    int numBoxes,
    int valuesPerBox,
    double scale,
    double padX,
    double padY,
    double originalWidth,
    double originalHeight,
  ) {
    final List<_Detection> preliminaryDetections = [];
    final int classes = valuesPerBox - 5;

    for (int i = 0; i < numBoxes; i++) {
      final int offset = i * valuesPerBox;
      final double xCenter = output[offset];
      final double yCenter = output[offset + 1];
      final double boxWidth = output[offset + 2];
      final double boxHeight = output[offset + 3];
      final double objectness = output[offset + 4];

      double bestClassScore = 0;
      int bestClassIndex = -1;
      for (int c = 0; c < classes; c++) {
        final double classScore = output[offset + 5 + c];
        if (classScore > bestClassScore) {
          bestClassScore = classScore;
          bestClassIndex = c;
        }
      }

      if (bestClassIndex < 0) {
        continue;
      }

      final double confidence = objectness * bestClassScore;
      if (confidence < _confidenceThreshold) {
        continue;
      }

      final double xMin = xCenter - boxWidth / 2;
      final double yMin = yCenter - boxHeight / 2;
      final double xMax = xCenter + boxWidth / 2;
      final double yMax = yCenter + boxHeight / 2;

      final double correctedXMin = ((xMin - padX) / scale).clamp(0.0, originalWidth);
      final double correctedYMin = ((yMin - padY) / scale).clamp(0.0, originalHeight);
      final double correctedXMax = ((xMax - padX) / scale).clamp(0.0, originalWidth);
      final double correctedYMax = ((yMax - padY) / scale).clamp(0.0, originalHeight);

      final Rect rect = Rect.fromLTRB(
        correctedXMin,
        correctedYMin,
        correctedXMax,
        correctedYMax,
      );

      final String label = bestClassIndex < _labels.length
          ? _labels[bestClassIndex]
          : 'Clase $bestClassIndex';

      final double distance = _estimateDistance(rect.height, originalHeight);

      preliminaryDetections.add(
        _Detection(
          boundingBox: rect,
          label: label,
          score: confidence,
          distanceMeters: distance,
        ),
      );
    }

    preliminaryDetections.sort((a, b) => b.score.compareTo(a.score));
    return _applyNms(preliminaryDetections);
  }

  double _estimateDistance(double boxHeight, double imageHeight) {
    final double relativeSize = (boxHeight / imageHeight).clamp(1e-6, 1.0);
    final double distance = (0.55 / relativeSize).clamp(0.3, 8.0);
    return double.parse(distance.toStringAsFixed(2));
  }

  List<_Detection> _applyNms(List<_Detection> detections) {
    final List<_Detection> result = [];
    for (final detection in detections) {
      bool shouldAdd = true;
      for (final kept in result) {
        if (detection.label != kept.label) {
          continue;
        }
        final double overlap = _iou(detection.boundingBox, kept.boundingBox);
        if (overlap > _iouThreshold) {
          shouldAdd = false;
          break;
        }
      }
      if (shouldAdd) {
        result.add(detection);
      }
      if (result.length >= 15) {
        break;
      }
    }
    return result;
  }

  double _iou(Rect a, Rect b) {
    final double areaA = a.width * a.height;
    final double areaB = b.width * b.height;
    if (areaA <= 0 || areaB <= 0) {
      return 0;
    }

    final double xMin = math.max(a.left, b.left);
    final double yMin = math.max(a.top, b.top);
    final double xMax = math.min(a.right, b.right);
    final double yMax = math.min(a.bottom, b.bottom);

    final double intersectionWidth = math.max(0, xMax - xMin);
    final double intersectionHeight = math.max(0, yMax - yMin);
    final double intersectionArea = intersectionWidth * intersectionHeight;

    final double unionArea = areaA + areaB - intersectionArea;
    if (unionArea <= 0) {
      return 0;
    }

    return intersectionArea / unionArea;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: const Text('Detección en vivo'),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_initializing) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.white70),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    if (!_modelLoaded) {
      return const Center(
        child: Text(
          'Cargando modelo de detección...',
          style: TextStyle(color: Colors.white70),
        ),
      );
    }

    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(
        child: Text(
          'Cámara no inicializada.',
          style: TextStyle(color: Colors.white70),
        ),
      );
    }

    final preview = Center(
      child: AspectRatio(
        aspectRatio: controller.value.aspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            CameraPreview(controller),
            if (_lastImageSize != null)
              CustomPaint(
                painter: _DetectionPainter(
                  detections: _detections,
                  imageSize: _lastImageSize!,
                ),
              ),
          ],
        ),
      ),
    );

    return Stack(
      children: [
        Positioned.fill(child: preview),
        Positioned(
          top: 24,
          left: 24,
          child: _StatusChip(
            active: _detections.isNotEmpty,
            processing: _isProcessingFrame,
            detections: _detections.length,
          ),
        ),
        Positioned(
          left: 16,
          right: 16,
          bottom: 24,
          child: _DetectionsPanel(
            detections: _detections,
          ),
        ),
      ],
    );
  }
}

class _PreprocessResult {
  const _PreprocessResult({
    required this.input,
    required this.scale,
    required this.padX,
    required this.padY,
  });

  final Float32List input;
  final double scale;
  final double padX;
  final double padY;
}

class _Detection {
  const _Detection({
    required this.boundingBox,
    required this.label,
    required this.score,
    required this.distanceMeters,
  });

  final Rect boundingBox;
  final String label;
  final double score;
  final double distanceMeters;
}

class _DetectionPainter extends CustomPainter {
  _DetectionPainter({
    required this.detections,
    required this.imageSize,
  });

  final List<_Detection> detections;
  final Size imageSize;

  @override
  void paint(Canvas canvas, Size size) {
    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;

    final Paint boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..color = Colors.lightBlueAccent;

    final Paint backgroundPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.black.withOpacity(0.55);

    for (final detection in detections) {
      final Rect rect = Rect.fromLTRB(
        detection.boundingBox.left * scaleX,
        detection.boundingBox.top * scaleY,
        detection.boundingBox.right * scaleX,
        detection.boundingBox.bottom * scaleY,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(rect, const Radius.circular(12)),
        boxPaint,
      );

      final String text =
          '${detection.label} ${(detection.score * 100).toStringAsFixed(1)}%\n~${detection.distanceMeters} m';
      final TextSpan span = TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );
      final TextPainter painter = TextPainter(
        text: span,
        textAlign: TextAlign.left,
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: size.width * 0.5);

      const double padding = 8;
      final Size labelSize = Size(
        painter.width + padding * 2,
        painter.height + padding * 2,
      );
      final double labelLeft = rect.left;
      final double labelTop = math.max(0, rect.top - labelSize.height - 6);

      final Rect labelRect = Rect.fromLTWH(
        labelLeft,
        labelTop,
        labelSize.width,
        labelSize.height,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(10)),
        backgroundPaint,
      );

      painter.paint(
        canvas,
        Offset(labelLeft + padding, labelTop + padding),
      );
    }
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) {
    return !listEquals(oldDelegate.detections, detections) ||
        oldDelegate.imageSize != imageSize;
  }
}

class _StatusChip extends StatelessWidget {
  const _StatusChip({
    required this.active,
    required this.processing,
    required this.detections,
  });

  final bool active;
  final bool processing;
  final int detections;

  @override
  Widget build(BuildContext context) {
    final Color color = active ? Colors.greenAccent : Colors.blueGrey;
    final String label = processing
        ? 'Analizando...'
        : active
            ? 'Objetos: $detections'
            : 'Escena sin objetos';

    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.6),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color, width: 1.6),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              active ? Icons.insights : Icons.visibility_outlined,
              color: color,
              size: 18,
            ),
            const SizedBox(width: 8),
            Text(
              label,
              style: const TextStyle(color: Colors.white),
            ),
          ],
        ),
      ),
    );
  }
}

class _DetectionsPanel extends StatelessWidget {
  const _DetectionsPanel({required this.detections});

  final List<_Detection> detections;

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) {
      return DecoratedBox(
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.6),
          borderRadius: BorderRadius.circular(16),
        ),
        child: const Padding(
          padding: EdgeInsets.all(16),
          child: Text(
            'Apunta la cámara a un objeto para comenzar a detectarlo.',
            style: TextStyle(color: Colors.white70),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    final theme = Theme.of(context);
    return DecoratedBox(
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.75),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Detecciones recientes',
              style: theme.textTheme.titleSmall?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            ...detections.take(4).map(
              (detection) => Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Row(
                  children: [
                    Container(
                      width: 10,
                      height: 10,
                      decoration: const BoxDecoration(
                        color: Colors.lightBlueAccent,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        detection.label,
                        style: const TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    Text(
                      '${(detection.score * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(color: Colors.white70),
                    ),
                    const SizedBox(width: 12),
                    Text(
                      '~${detection.distanceMeters.toStringAsFixed(1)} m',
                      style: const TextStyle(color: Colors.white70),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

const List<String> _cocoLabels = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
];
