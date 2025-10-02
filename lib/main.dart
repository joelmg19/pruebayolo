import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

import 'detection_page.dart';

void main() {
  runApp(const CellSayApp());
}

class CellSayApp extends StatelessWidget {
  const CellSayApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'CellSay',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final FlutterTts flutterTts = FlutterTts();
  final stt.SpeechToText _speechToText = stt.SpeechToText();

  double _textScale = 1.2;
  double _voiceRate = 0.6;
  double _voicePitch = 1.1;
  bool _speechAvailable = false;
  bool _isListening = false;
  bool _deteccionObstaculosActiva = true;
  bool _deteccionSemaforosActiva = true;
  bool _peligroMovimientoDetectado = false;
  DateTime? _inicioRuta;
  String _ultimoComando = '---';
  String? _speechStatus;
  String? _speechError;

  static const _prefsKey = 'cellsay_text_scale';
  static const _prefsVoiceRate = 'cellsay_voice_rate';
  static const _prefsVoicePitch = 'cellsay_voice_pitch';

  @override
  void initState() {
    super.initState();
    _configureTts();
    _loadScale();
    _loadVoiceSettings();
    _initSpeech();
  }

  @override
  void dispose() {
    flutterTts.stop();
    _speechToText.stop();
    super.dispose();
  }

  Future<void> _loadScale() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _textScale = prefs.getDouble(_prefsKey) ?? 1.2;
    });
  }

  Future<void> _saveScale(double value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_prefsKey, value);
  }

  Future<void> _configureTts() async {
    await flutterTts.setLanguage('es-ES');
    await flutterTts.setSpeechRate(_voiceRate);
    await flutterTts.setPitch(_voicePitch);
  }

  Future<void> _loadVoiceSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _voiceRate = prefs.getDouble(_prefsVoiceRate) ?? _voiceRate;
      _voicePitch = prefs.getDouble(_prefsVoicePitch) ?? _voicePitch;
    });
    await flutterTts.setSpeechRate(_voiceRate);
    await flutterTts.setPitch(_voicePitch);
  }

  Future<void> _saveVoiceSettings() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble(_prefsVoiceRate, _voiceRate);
    await prefs.setDouble(_prefsVoicePitch, _voicePitch);
  }

  Future<void> _initSpeech() async {
    final available = await _speechToText.initialize(
      onStatus: (status) {
        setState(() => _speechStatus = status);
      },
      onError: (error) {
        setState(() => _speechError = error.errorMsg);
      },
    );

    setState(() {
      _speechAvailable = available;
    });
  }

  Future<void> _speak(String texto) async {
    await flutterTts.setSpeechRate(_voiceRate);
    await flutterTts.setPitch(_voicePitch);
    await flutterTts.speak(texto);
  }

  Future<void> _decirHora() async {
    final ahora = DateTime.now();
    final horaFormateada = DateFormat('HH:mm').format(ahora);
    await _speak('La hora actual es $horaFormateada');
  }

  String _formatearHora(DateTime? fecha) {
    if (fecha == null) return 'No iniciada';
    return DateFormat('HH:mm').format(fecha);
  }

  Future<void> _startListening() async {
    if (!_speechAvailable) {
      await _initSpeech();
    }

    if (!_speechAvailable) {
      await _speak('No pude activar el reconocimiento de voz.');
      return;
    }

    setState(() {
      _isListening = true;
      _ultimoComando = 'Escuchando...';
    });

    await _speechToText.listen(
      localeId: 'es_ES',
      // Arreglo principal: 'command' ya no existe
      listenMode: stt.ListenMode.search, // usar search/dictation/deviceDefault según tu caso
      onResult: (result) {
        if (!mounted) return;
        final texto = result.recognizedWords.trim();
        if (texto.isEmpty) return;
        setState(() {
          _ultimoComando = texto;
        });
        if (result.finalResult) {
          _procesarComando(texto.toLowerCase());
          _stopListening();
        }
      },
    );
  }

  Future<void> _stopListening() async {
    await _speechToText.stop();
    if (mounted) {
      setState(() {
        _isListening = false;
      });
    }
  }

  Future<void> _procesarComando(String comando) async {
    comando = comando.toLowerCase();

    if (comando.contains('hora')) {
      await _decirHora();
      return;
    }

    if (comando.contains('iniciar ruta')) {
      _iniciarRuta();
      await _speak('Ruta iniciada');
      return;
    }

    if (comando.contains('detener ruta') || comando.contains('finalizar ruta')) {
      _detenerRuta();
      await _speak('Ruta detenida');
      return;
    }

    if (comando.contains('activar obst')) {
      setState(() => _deteccionObstaculosActiva = true);
      await _speak('Detección de obstáculos activada');
      return;
    }

    if (comando.contains('desactivar obst')) {
      setState(() => _deteccionObstaculosActiva = false);
      await _speak('Detección de obstáculos desactivada');
      return;
    }

    if (comando.contains('activar sem')) {
      setState(() => _deteccionSemaforosActiva = true);
      await _speak('Reconocimiento de semáforos activado');
      return;
    }

    if (comando.contains('desactivar sem')) {
      setState(() => _deteccionSemaforosActiva = false);
      await _speak('Reconocimiento de semáforos desactivado');
      return;
    }

    if (comando.contains('peligro') && comando.contains('detect')) {
      setState(() => _peligroMovimientoDetectado = true);
      await _speak('Peligro en movimiento detectado');
      return;
    }

    if (comando.contains('limpiar peligro') || comando.contains('sin peligro')) {
      setState(() => _peligroMovimientoDetectado = false);
      await _speak('Peligro en movimiento despejado');
      return;
    }

    await _speak('No entendí el comando $comando');
  }

  void _iniciarRuta() {
    setState(() {
      _inicioRuta = DateTime.now();
    });
  }

  void _detenerRuta() {
    setState(() {
      _inicioRuta = null;
    });
  }

  Future<void> _abrirAjustesVoz() async {
    double tempRate = _voiceRate;
    double tempPitch = _voicePitch;

    await showModalBottomSheet(
      context: context,
      showDragHandle: true,
      backgroundColor: Theme.of(context).colorScheme.surface,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setModalState) {
            return Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    'Configuración de voz',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  _SliderRow(
                    label: 'Velocidad',
                    value: tempRate,
                    min: 0.3,
                    max: 1.0,
                    divisions: 7,
                    onChanged: (value) {
                      setModalState(() => tempRate = value);
                    },
                  ),
                  _SliderRow(
                    label: 'Tono',
                    value: tempPitch,
                    min: 0.7,
                    max: 1.5,
                    divisions: 8,
                    onChanged: (value) {
                      setModalState(() => tempPitch = value);
                    },
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('Cancelar'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton(
                          onPressed: () async {
                            setState(() {
                              _voiceRate = tempRate;
                              _voicePitch = tempPitch;
                            });
                            await _saveVoiceSettings();
                            await flutterTts.setSpeechRate(_voiceRate);
                            await flutterTts.setPitch(_voicePitch);
                            if (context.mounted) Navigator.pop(context);
                          },
                          child: const Text('Aplicar'),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }

  void _abrirAjustesTexto() {
    double temp = _textScale;

    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      isScrollControlled: false,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (ctx, setModalState) {
            return Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text(
                    'Tamaño de fuente',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      const Text('A-'),
                      Expanded(
                        child: Slider(
                          value: temp,
                          min: 0.8,
                          max: 2.0,
                          divisions: 12,
                          label: temp.toStringAsFixed(1),
                          onChanged: (v) {
                            setModalState(() {
                              temp = v;
                            });
                          },
                        ),
                      ),
                      const Text('A+'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    children: [
                      _ChipPreset('Pequeña', 0.9, (v) {
                        setModalState(() => temp = v);
                      }),
                      _ChipPreset('Normal', 1.0, (v) {
                        setModalState(() => temp = v);
                      }),
                      _ChipPreset('Grande', 1.3, (v) {
                        setModalState(() => temp = v);
                      }),
                      _ChipPreset('Enorme', 1.7, (v) {
                        setModalState(() => temp = v);
                      }),
                    ],
                  ),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton(
                          onPressed: () => Navigator.pop(context),
                          child: const Text('Cancelar'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton(
                          onPressed: () async {
                            setState(() => _textScale = temp);
                            await _saveScale(_textScale);
                            if (context.mounted) Navigator.pop(context);
                          },
                          child: const Text('Aplicar'),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                ],
              ),
            );
          },
        );
      },
    );
  }

  Future<void> _abrirDeteccionObjetos() async {
    await Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const DetectionPage()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;

    return MediaQuery(
      data: MediaQuery.of(context).copyWith(textScaler: TextScaler.linear(_textScale)),
      child: Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(
          backgroundColor: Colors.black,
          foregroundColor: Colors.white,
          title: const Text('CellSay'),
          actions: [
            IconButton(
              tooltip: 'Configuración de voz',
              onPressed: _abrirAjustesVoz,
              icon: const Icon(Icons.record_voice_over),
            ),
            IconButton(
              tooltip: 'Tamaño de fuente',
              onPressed: _abrirAjustesTexto,
              icon: const Icon(Icons.text_fields),
            ),
          ],
        ),
        body: SafeArea(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'Asistente de movilidad en tiempo real',
                  style: textTheme.headlineSmall?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                Text(
                  'Controla la detección por voz y recibe avisos de obstáculos, semáforos y peligros en movimiento utilizando el modelo YOLO11n.',
                  style: textTheme.bodyMedium?.copyWith(color: Colors.white70),
                ),
                const SizedBox(height: 20),
                _StatusCard(
                  title: 'Detección de obstáculo cercano',
                  description: 'Avisos sonoros inmediatos cuando se detecta un objeto frente a ti.',
                  icon: Icons.sensors,
                  active: _deteccionObstaculosActiva,
                  onToggle: (value) async {
                    setState(() => _deteccionObstaculosActiva = value);
                    await _speak(value
                        ? 'Detección de obstáculos activada'
                        : 'Detección de obstáculos desactivada');
                  },
                ),
                const SizedBox(height: 12),
                _StatusCard(
                  title: 'Reconocimiento de semáforo',
                  description: 'Identifica el color del semáforo y anuncia cuándo cruzar.',
                  icon: Icons.traffic,
                  active: _deteccionSemaforosActiva,
                  onToggle: (value) async {
                    setState(() => _deteccionSemaforosActiva = value);
                    await _speak(value
                        ? 'Reconocimiento de semáforos activado'
                        : 'Reconocimiento de semáforos desactivado');
                  },
                ),
                const SizedBox(height: 12),
                _DetectionFeatureCard(
                  onOpen: _abrirDeteccionObjetos,
                ),
                const SizedBox(height: 12),
                _DangerCard(
                  active: _peligroMovimientoDetectado,
                  onReset: () async {
                    setState(() => _peligroMovimientoDetectado = false);
                    await _speak('Peligro despejado');
                  },
                  onSimulate: () async {
                    setState(() => _peligroMovimientoDetectado = true);
                    await _speak('Atención. Peligro en movimiento detectado');
                  },
                ),
                const SizedBox(height: 12),
                _RouteCard(
                  inicioRuta: _inicioRuta,
                  horaInicio: _formatearHora(_inicioRuta),
                  onStart: () async {
                    _iniciarRuta();
                    await _speak('Ruta iniciada a las ${_formatearHora(_inicioRuta)}');
                  },
                  onStop: () async {
                    _detenerRuta();
                    await _speak('Ruta detenida');
                  },
                ),
                const SizedBox(height: 12),
                _VoiceControlCard(
                  isListening: _isListening,
                  speechAvailable: _speechAvailable,
                  ultimoComando: _ultimoComando,
                  speechStatus: _speechStatus,
                  speechError: _speechError,
                  onStartListening: _startListening,
                  onStopListening: _stopListening,
                ),
                const SizedBox(height: 12),
                ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blueAccent,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                  ),
                  onPressed: _decirHora,
                  icon: const Icon(Icons.access_time),
                  label: const Text('Decir hora actual'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _ChipPreset extends StatelessWidget {
  final String label;
  final double value;
  final void Function(double) onPick;

  const _ChipPreset(this.label, this.value, this.onPick);

  @override
  Widget build(BuildContext context) {
    return ActionChip(
      label: Text(label),
      onPressed: () => onPick(value),
    );
  }
}

class _SliderRow extends StatelessWidget {
  final String label;
  final double value;
  final double min;
  final double max;
  final int divisions;
  final ValueChanged<double> onChanged;

  const _SliderRow({
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.divisions,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold),
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          onChanged: onChanged,
          label: value.toStringAsFixed(2),
        ),
      ],
    );
  }
}

class _StatusCard extends StatelessWidget {
  final String title;
  final String description;
  final IconData icon;
  final bool active;
  final ValueChanged<bool> onToggle;

  const _StatusCard({
    required this.title,
    required this.description,
    required this.icon,
    required this.active,
    required this.onToggle,
  });

  @override
  Widget build(BuildContext context) {
    return _BaseCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: Colors.white),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  title,
                  style: Theme.of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
              Switch(
                value: active,
                onChanged: onToggle,
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            description,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white70),
          ),
        ],
      ),
    );
  }
}

class _DetectionFeatureCard extends StatelessWidget {
  final VoidCallback onOpen;

  const _DetectionFeatureCard({required this.onOpen});

  @override
  Widget build(BuildContext context) {
    return _BaseCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.camera_alt, color: Colors.white),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Detección con cámara en vivo',
                  style: Theme.of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Abre la cámara para detectar objetos en tiempo real con YOLO11n y recibir una estimación de distancia.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white70),
          ),
          const SizedBox(height: 12),
          ElevatedButton.icon(
            onPressed: onOpen,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.blueAccent,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 16),
            ),
            icon: const Icon(Icons.play_arrow),
            label: const Text('Abrir cámara de detección'),
          ),
          const SizedBox(height: 4),
          Text(
            'Las distancias son aproximadas y dependen del tamaño del objeto en pantalla.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white54),
          ),
        ],
      ),
    );
  }
}

class _DangerCard extends StatelessWidget {
  final bool active;
  final Future<void> Function() onSimulate;
  final Future<void> Function() onReset;

  const _DangerCard({
    required this.active,
    required this.onSimulate,
    required this.onReset,
  });

  @override
  Widget build(BuildContext context) {
    return _BaseCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                active ? Icons.warning_rounded : Icons.shield_outlined,
                color: active ? Colors.orangeAccent : Colors.white,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Peligro en movimiento',
                  style: Theme.of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
              Container(
                decoration: BoxDecoration(
                  color: active ? Colors.orangeAccent : Colors.green,
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                child: Text(
                  active ? 'Detectado' : 'Seguro',
                  style: const TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            active
                ? 'Se detectó movimiento peligroso cerca. Toma precauciones inmediatas.'
                : 'Sin peligros cercanos detectados. Mantente atento a las alertas.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white70),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: onReset,
                  child: const Text('Marcar despejado'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton(
                  onPressed: onSimulate,
                  child: const Text('Simular peligro'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _RouteCard extends StatelessWidget {
  final DateTime? inicioRuta;
  final String horaInicio;
  final Future<void> Function() onStart;
  final Future<void> Function() onStop;

  const _RouteCard({
    required this.inicioRuta,
    required this.horaInicio,
    required this.onStart,
    required this.onStop,
  });

  @override
  Widget build(BuildContext context) {
    final rutaActiva = inicioRuta != null;

    return _BaseCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                rutaActiva ? Icons.directions_walk : Icons.flag_outlined,
                color: Colors.white,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Ruta asistida',
                  style: Theme.of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
              Container(
                decoration: BoxDecoration(
                  color: rutaActiva ? Colors.lightBlueAccent : Colors.grey.shade700,
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                child: Text(
                  rutaActiva ? 'Activa' : 'Inactiva',
                  style: const TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            rutaActiva
                ? 'Ruta iniciada a las $horaInicio. Recibirás avisos de obstáculos mientras caminas.'
                : 'Presiona iniciar para comenzar una ruta guiada con alertas en tiempo real.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white70),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: rutaActiva
                    ? OutlinedButton(
                  onPressed: onStop,
                  child: const Text('Detener ruta'),
                )
                    : ElevatedButton(
                  onPressed: onStart,
                  child: const Text('Iniciar ruta'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _VoiceControlCard extends StatelessWidget {
  final bool isListening;
  final bool speechAvailable;
  final String ultimoComando;
  final String? speechStatus;
  final String? speechError;
  final Future<void> Function() onStartListening;
  final Future<void> Function() onStopListening;

  const _VoiceControlCard({
    required this.isListening,
    required this.speechAvailable,
    required this.ultimoComando,
    required this.speechStatus,
    required this.speechError,
    required this.onStartListening,
    required this.onStopListening,
  });

  @override
  Widget build(BuildContext context) {
    return _BaseCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.mic, color: isListening ? Colors.redAccent : Colors.white),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Control por voz en español',
                  style: Theme.of(context)
                      .textTheme
                      .titleMedium
                      ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
              Switch(
                value: isListening,
                onChanged: (_) {
                  if (isListening) {
                    onStopListening();
                  } else {
                    onStartListening();
                  }
                },
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            speechAvailable
                ? 'Di: "iniciar ruta", "activar obstáculos", "decir hora" o "reconocer semáforos".'
                : 'El reconocimiento de voz no está disponible en este dispositivo.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white70),
          ),
          const SizedBox(height: 12),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Último comando:',
                  style: Theme.of(context)
                      .textTheme
                      .bodyMedium
                      ?.copyWith(color: Colors.white, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 6),
                Text(
                  ultimoComando,
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(color: Colors.white),
                ),
                if (speechStatus != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Estado: $speechStatus',
                    style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.white70),
                  ),
                ],
                if (speechError != null) ...[
                  const SizedBox(height: 8),
                  Text(
                    'Error: $speechError',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.redAccent, fontWeight: FontWeight.bold),
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: onStopListening,
                  icon: const Icon(Icons.stop),
                  label: const Text('Detener escucha'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: onStartListening,
                  icon: const Icon(Icons.mic_none),
                  label: const Text('Escuchar comando'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _BaseCard extends StatelessWidget {
  final Widget child;

  const _BaseCard({required this.child});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: const Color(0xFF1A1A1A),
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: child,
      ),
    );
  }
}
