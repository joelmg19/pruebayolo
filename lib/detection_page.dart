import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage>
    with WidgetsBindingObserver {
  static const int _inputSize = 640;
  static const double _confidenceThreshold = 0.35;
  static const double _nmsThreshold = 0.45;
  static const double _assumedObjectHeightMeters = 1.7;
  static const double _assumedFocalLengthPixels = 880;

  final List<String> _labels = const [
    'persona',
    'bicicleta',
    'auto',
    'motocicleta',
    'avión',
    'autobús',
    'tren',
    'camión',
    'barco',
    'semaforo',
    'hidrante',
    'señal de pare',
    'parquímetro',
    'banca',
    'ave',
    'gato',
    'perro',
    'caballo',
    'oveja',
    'vaca',
    'elefante',
    'oso',
    'cebra',
    'jirafa',
    'mochila',
    'paraguas',
    'bolso',
    'corbata',
    'maleta',
    'frisbee',
    'esquíes',
    'snowboard',
    'balón deportivo',
    'cometa',
    'bate de béisbol',
    'guante de béisbol',
    'patineta',
    'tabla de surf',
    'raqueta de tenis',
    'botella',
    'copa de vino',
    'taza',
    'tenedor',
    'cuchillo',
    'cuchara',
    'tazón',
    'plátano',
    'manzana',
    'sándwich',
    'naranja',
    'brócoli',
    'zanahoria',
    'hot dog',
    'pizza',
    'donut',
    'pastel',
    'silla',
    'sofá',
    'planta en maceta',
    'cama',
    'mesa',
    'inodoro',
    'televisor',
    'laptop',
    'ratón',
    'control remoto',
    'teclado',
    'teléfono celular',
    'microondas',
    'horno',
    'tostadora',
    'lavamanos',
    'refrigerador',
    'libro',
    'reloj',
    'florero',
    'tijeras',
    'oso de peluche',
    'secador de cabello',
    'cepillo de dientes',
  ];

  CameraController? _cameraController;
  Interpreter? _interpreter;
  List<DetectionResult> _results = const [];
  bool _isBusy = false;
  bool _initializing = true;
  String? _error;
  Size _previewSize = Size.zero;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initialize();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopImageStream();
    _cameraController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive || state == AppLifecycleState.paused) {
      _stopImageStream();
      _cameraController?.dispose();
      _cameraController = null;
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  Future<void> _initialize() async {
    try {
      await _loadModel();
      await _initializeCamera();
      if (mounted) {
        setState(() {
          _initializing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'No se pudo inicializar la detección: $e';
          _initializing = false;
        });
      }
    }
  }

  Future<void> _loadModel() async {
    final options = InterpreterOptions()..threads = 2;
    _interpreter = await Interpreter.fromAsset('models/yolo11n.tflite', options: options);
  }

  Future<void> _initializeCamera() async {
    try {
      await _cameraController?.dispose();

      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw Exception('No se encontró ninguna cámara disponible.');
      }
      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();
      final previewSize = controller.value.previewSize;
      if (previewSize != null) {
        _previewSize = Size(previewSize.height, previewSize.width);
      } else {
        _previewSize = Size(_inputSize.toDouble(), _inputSize.toDouble());
      }
      await controller.startImageStream(_processCameraImage);
      if (mounted) {
        setState(() {
          _cameraController = controller;
        });
      } else {
        await controller.dispose();
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Error al iniciar la cámara: $e';
        });
      }
    }
  }

  Future<void> _stopImageStream() async {
    if (_cameraController != null && _cameraController!.value.isStreamingImages) {
      await _cameraController!.stopImageStream();
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_interpreter == null || _isBusy) {
      return;
    }
    _isBusy = true;

    try {
      final imageLib = _convertYUV420ToImage(image);
      final resizedImage = image_lib.copyResize(
        imageLib,
        width: _inputSize,
        height: _inputSize,
        interpolation: image_lib.Interpolation.linear,
      );
      final inputBuffer = _convertImageToFloat32(resizedImage);

      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputBuffer = _createOutputBuffer(outputTensor);

      _interpreter!.run(inputBuffer.buffer, outputBuffer.buffer);

      final detections = _parseDetections(
        _extractOutputData(outputBuffer, outputTensor.type),
        outputTensor.shape,
      );
      if (mounted) {
        setState(() {
          _results = detections;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = 'Error al procesar imagen: $e';
        });
      }
    } finally {
      _isBusy = false;
    }
  }

  List<DetectionResult> _parseDetections(List<double> outputs, List<int> outputShape) {
    int numDetections;
    int valuesPerDetection;
    if (outputShape.length == 3) {
      numDetections = outputShape[1];
      valuesPerDetection = outputShape[2];
    } else if (outputShape.length == 2) {
      numDetections = outputShape[0];
      valuesPerDetection = outputShape[1];
    } else {
      return const [];
    }

    final List<DetectionResult> parsedDetections = [];
    for (int i = 0; i < numDetections; i++) {
      final offset = i * valuesPerDetection;
      if (offset + 5 >= outputs.length) {
        break;
      }

      double x = outputs[offset];
      double y = outputs[offset + 1];
      double w = outputs[offset + 2];
      double h = outputs[offset + 3];
      final double objectConfidence = outputs[offset + 4];

      if (objectConfidence < _confidenceThreshold) {
        continue;
      }

      final classScores = outputs.sublist(offset + 5, offset + valuesPerDetection);
      if (classScores.isEmpty) {
        continue;
      }

      double maxScore = classScores.first;
      int maxIndex = 0;
      for (int j = 1; j < classScores.length; j++) {
        if (classScores[j] > maxScore) {
          maxScore = classScores[j];
          maxIndex = j;
        }
      }

      final double confidence = objectConfidence * maxScore;
      if (confidence < _confidenceThreshold) {
        continue;
      }

      final bool normalized = x <= 1.0 && y <= 1.0 && w <= 1.0 && h <= 1.0;
      final double boxScale = normalized ? 1.0 : _inputSize.toDouble();

      x *= boxScale;
      y *= boxScale;
      w *= boxScale;
      h *= boxScale;

      final double left = ((x - w / 2) / _inputSize).clamp(0.0, 1.0);
      final double top = ((y - h / 2) / _inputSize).clamp(0.0, 1.0);
      final double right = ((x + w / 2) / _inputSize).clamp(0.0, 1.0);
      final double bottom = ((y + h / 2) / _inputSize).clamp(0.0, 1.0);

      final rect = Rect.fromLTRB(left, top, right, bottom);
      final distanceMeters = _estimateDistance(rect.height);
      final label = maxIndex < _labels.length ? _labels[maxIndex] : 'objeto';

      parsedDetections.add(DetectionResult(
        boundingBox: rect,
        label: label,
        confidence: confidence,
        distanceMeters: distanceMeters,
      ));
    }

    return _nonMaxSuppression(parsedDetections, _nmsThreshold).take(10).toList();
  }

  double _estimateDistance(double normalizedHeight) {
    if (normalizedHeight <= 0) {
      return double.infinity;
    }
    final double pixelHeight = normalizedHeight * _inputSize;
    final double distanceMeters =
        (_assumedObjectHeightMeters * _assumedFocalLengthPixels) / (pixelHeight + 1e-6);
    return distanceMeters.isFinite ? distanceMeters : double.infinity;
  }

  List<DetectionResult> _nonMaxSuppression(List<DetectionResult> detections, double iouThreshold) {
    final List<DetectionResult> sortedDetections = List.of(detections)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    final List<DetectionResult> kept = [];

    for (final candidate in sortedDetections) {
      bool shouldAdd = true;
      for (final keptDetection in kept) {
        if (_intersectionOverUnion(candidate.boundingBox, keptDetection.boundingBox) > iouThreshold) {
          shouldAdd = false;
          break;
        }
      }
      if (shouldAdd) {
        kept.add(candidate);
      }
    }
    return kept;
  }

  double _intersectionOverUnion(Rect a, Rect b) {
    final double intersectionWidth = math.max(0, math.min(a.right, b.right) - math.max(a.left, b.left));
    final double intersectionHeight = math.max(0, math.min(a.bottom, b.bottom) - math.max(a.top, b.top));
    final double intersectionArea = intersectionWidth * intersectionHeight;
    if (intersectionArea <= 0) {
      return 0;
    }
    final double areaA = a.width * a.height;
    final double areaB = b.width * b.height;
    final double unionArea = areaA + areaB - intersectionArea;
    return unionArea <= 0 ? 0 : intersectionArea / unionArea;
  }

  Float32List _convertImageToFloat32(image_lib.Image image) {
    final int width = image.width;
    final int height = image.height;
    final Float32List buffer = Float32List(width * height * 3);
    int bufferIndex = 0;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final image_lib.Pixel pixel = image.getPixel(x, y);
        buffer[bufferIndex++] = (pixel.r / 255.0).toDouble();
        buffer[bufferIndex++] = (pixel.g / 255.0).toDouble();
        buffer[bufferIndex++] = (pixel.b / 255.0).toDouble();
      }
    }

    return buffer;
  }

  TypedData _createOutputBuffer(Tensor outputTensor) {
    final int elementCount = outputTensor.shape.fold<int>(1, (value, element) => value * element);
    switch (outputTensor.type) {
      case TensorType.float32:
        return Float32List(elementCount);
      case TensorType.int32:
        return Int32List(elementCount);
      case TensorType.uint8:
        return Uint8List(elementCount);
      default:
        throw UnsupportedError('Tipo de tensor no soportado: ${outputTensor.type}');
    }
  }

  List<double> _extractOutputData(TypedData buffer, TensorType type) {
    if (buffer is Float32List) {
      return List<double>.generate(buffer.length, (index) => buffer[index].toDouble());
    }
    if (buffer is Int32List) {
      return List<double>.generate(buffer.length, (index) => buffer[index].toDouble());
    }
    if (buffer is Uint8List) {
      return List<double>.generate(buffer.length, (index) => buffer[index].toDouble());
    }
    throw UnsupportedError('Tipo de datos de salida no soportado: $type');
  }

  image_lib.Image _convertYUV420ToImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final image_lib.Image img = image_lib.Image(width: width, height: height);
    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final Uint8List bytesY = planeY.bytes;
    final Uint8List bytesU = planeU.bytes;
    final Uint8List bytesV = planeV.bytes;

    final int uvRowStride = planeU.bytesPerRow;
    final int uvPixelStride = planeU.bytesPerPixel ?? 1;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = (y ~/ 2) * uvRowStride + (x ~/ 2) * uvPixelStride;
        final int index = y * width + x;

        final int yp = bytesY[index];
        final int up = bytesU[uvIndex];
        final int vp = bytesV[uvIndex];

        int r = (yp + (vp - 128) * 1436 / 1024).round();
        int g = (yp - (vp - 128) * 731 / 1024 - (up - 128) * 354 / 1024).round();
        int b = (yp + (up - 128) * 1814 / 1024).round();

        r = r.clamp(0, 255);
        g = g.clamp(0, 255);
        b = b.clamp(0, 255);

        img.setPixelRgba(x, y, r, g, b, 255);
      }
    }

    return img;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Detección en vivo'),
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_initializing) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Text(
            _error!,
            style: const TextStyle(color: Colors.white70),
            textAlign: TextAlign.center,
          ),
        ),
      );
    }

    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return const Center(
        child: Text(
          'Inicializando cámara...',
          style: TextStyle(color: Colors.white70),
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        final previewSize = _previewSize;
        final double previewWidth =
            previewSize.width == 0 ? _inputSize.toDouble() : previewSize.width;
        final double previewHeight =
            previewSize.height == 0 ? _inputSize.toDouble() : previewSize.height;

        return Column(
          children: [
            Expanded(
              child: Stack(
                fit: StackFit.expand,
                children: [
                  FittedBox(
                    fit: BoxFit.cover,
                    child: SizedBox(
                      width: previewWidth,
                      height: previewHeight,
                      child: CameraPreview(_cameraController!),
                    ),
                  ),
                  Positioned.fill(
                    child: CustomPaint(
                      painter: _DetectionPainter(
                        detections: _results,
                      ),
                    ),
                  ),
                  Positioned(
                    left: 16,
                    top: 16,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.black.withOpacity(0.5),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '${_results.length} objetos detectados',
                        style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Container(
              width: double.infinity,
              decoration: const BoxDecoration(
                color: Color(0xFF111111),
                border: Border(top: BorderSide(color: Colors.white12)),
              ),
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 24),
              child: _results.isEmpty
                  ? const Text(
                      'Apunta la cámara hacia un objeto para comenzar la detección.',
                      style: TextStyle(color: Colors.white70),
                    )
                  : Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Objetos detectados',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 12),
                        ..._results.map((detection) {
                          return Padding(
                            padding: const EdgeInsets.symmetric(vertical: 6),
                            child: Row(
                              children: [
                                Expanded(
                                  child: Text(
                                    _formatDetectionLabel(detection),
                                    style: const TextStyle(color: Colors.white),
                                  ),
                                ),
                                Text(
                                  _formatDistance(detection.distanceMeters),
                                  style: const TextStyle(color: Colors.lightBlueAccent, fontWeight: FontWeight.bold),
                                ),
                              ],
                            ),
                          );
                        }),
                      ],
                    ),
            ),
          ],
        );
      },
    );
  }

  String _formatDetectionLabel(DetectionResult detection) {
    final confidence = (detection.confidence * 100).clamp(0, 100).toStringAsFixed(0);
    return '${detection.label} ($confidence%)';
  }

  String _formatDistance(double distance) {
    if (!distance.isFinite) {
      return 'distancia desconocida';
    }
    if (distance > 20) {
      return '>20 m';
    }
    if (distance >= 5) {
      return '${distance.toStringAsFixed(0)} m';
    }
    return '${distance.toStringAsFixed(1)} m';
  }
}

class DetectionResult {
  final Rect boundingBox;
  final String label;
  final double confidence;
  final double distanceMeters;

  const DetectionResult({
    required this.boundingBox,
    required this.label,
    required this.confidence,
    required this.distanceMeters,
  });
}

class _DetectionPainter extends CustomPainter {
  _DetectionPainter({required this.detections});

  final List<DetectionResult> detections;

  @override
  void paint(Canvas canvas, Size size) {
    final Paint boxPaint = Paint()
      ..color = Colors.lightBlueAccent.withOpacity(0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.lightBlueAccent.withOpacity(0.15)
      ..style = PaintingStyle.fill;

    for (final detection in detections) {
      final Rect rect = Rect.fromLTRB(
        detection.boundingBox.left * size.width,
        detection.boundingBox.top * size.height,
        detection.boundingBox.right * size.width,
        detection.boundingBox.bottom * size.height,
      );

      canvas.drawRect(rect, fillPaint);
      canvas.drawRect(rect, boxPaint);

      final textSpan = TextSpan(
        text:
            '${detection.label} ${(detection.confidence * 100).toStringAsFixed(0)}%\n${_formatDistance(detection.distanceMeters)}',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
        maxLines: 2,
      )..layout(maxWidth: size.width * 0.6);

      final labelOffset = Offset(rect.left, math.max(0, rect.top - textPainter.height - 4));
      final backgroundRect = Rect.fromLTWH(
        labelOffset.dx,
        labelOffset.dy,
        textPainter.width + 12,
        textPainter.height + 8,
      );
      final backgroundPaint = Paint()
        ..color = Colors.black.withOpacity(0.6)
        ..style = PaintingStyle.fill;
      canvas.drawRRect(
        RRect.fromRectAndRadius(backgroundRect, const Radius.circular(8)),
        backgroundPaint,
      );
      textPainter.paint(canvas, labelOffset + const Offset(6, 4));
    }
  }

  String _formatDistance(double distance) {
    if (!distance.isFinite) {
      return 'dist. ?';
    }
    if (distance > 20) {
      return '>20 m';
    }
    if (distance >= 5) {
      return '${distance.toStringAsFixed(0)} m';
    }
    return '${distance.toStringAsFixed(1)} m';
  }

  @override
  bool shouldRepaint(covariant _DetectionPainter oldDelegate) {
    return oldDelegate.detections != detections;
  }
}
