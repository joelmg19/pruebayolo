import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:shared_preferences/shared_preferences.dart';

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

  double _textScale = 1.2;
  static const _prefsKey = 'cellsay_text_scale';

  @override
  void initState() {
    super.initState();
    _loadScale();
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

  Future<void> _decirHora() async {
    final ahora = DateTime.now();
    final horaFormateada = DateFormat('HH:mm').format(ahora);
    await flutterTts.speak("La hora actual es $horaFormateada");
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

  @override
  Widget build(BuildContext context) {
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
              tooltip: 'Tamaño de fuente',
              onPressed: _abrirAjustesTexto,
              icon: const Icon(Icons.text_fields),
            ),
          ],
        ),
        body: Center(
          child: ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.white,
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 20),
              minimumSize: const Size(200, 64),
            ),
            onPressed: _decirHora,
            child: const Text(
              'Decir hora',
              style: TextStyle(fontSize: 20),
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
