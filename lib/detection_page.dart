import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// Pantalla simplificada de detección.
///
/// El paquete `tflite_flutter` utilizado originalmente por la aplicación
/// dejó de compilar con las versiones recientes de Dart al intentar acceder
/// a `UnmodifiableUint8ListView`. Para evitar que toda la aplicación quede
/// inutilizable, se sustituyó la lógica de detección por una vista que
/// únicamente muestra la previsualización de la cámara junto con un mensaje
/// informativo.
///
/// Esta implementación mantiene la estructura de navegación existente y
/// permite seguir ampliando la funcionalidad en el futuro cuando exista un
/// motor de detección compatible.
class DetectionPage extends StatefulWidget {
  const DetectionPage({super.key});

  @override
  State<DetectionPage> createState() => _DetectionPageState();
}

class _DetectionPageState extends State<DetectionPage>
    with WidgetsBindingObserver {
  CameraController? _cameraController;
  bool _initializing = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    unawaited(_initializeCamera());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    unawaited(_stopPreview());
    _cameraController?.dispose();
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

  Future<void> _initializeCamera() async {
    try {
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
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = 'No se pudo iniciar la cámara: $e';
        _initializing = false;
      });
    }
  }

  Future<void> _startPreview() async {
    final controller = _cameraController;
    if (controller == null) return;

    if (!controller.value.isStreamingImages) {
      try {
        await controller.startImageStream((_) {});
      } catch (_) {
        // Ignoramos el error porque la previsualización puede seguir
        // funcionando incluso sin el stream.
      }
    }
  }

  Future<void> _stopPreview() async {
    final controller = _cameraController;
    if (controller == null) return;

    if (controller.value.isStreamingImages) {
      try {
        await controller.stopImageStream();
      } catch (_) {
        // No es crítico si falla al detener el stream.
      }
    }
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

    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(
        child: Text(
          'Cámara no inicializada.',
          style: TextStyle(color: Colors.white70),
        ),
      );
    }

    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(controller),
        Container(
          alignment: Alignment.bottomCenter,
          padding: const EdgeInsets.all(16),
          child: DecoratedBox(
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.65),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Padding(
              padding: EdgeInsets.all(16),
              child: Text(
                'La detección automática de objetos no está disponible en esta '
                'versión. Puedes utilizar la cámara como vista previa mientras '
                'se prepara una nueva integración de visión por computadora.',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  height: 1.4,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),
      ],
    );
  }
}
