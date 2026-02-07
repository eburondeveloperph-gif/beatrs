import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';
import { ConnectionStatus, TranscriptionPart } from './types';
import { decode, encode, decodeAudioData } from './services/audioUtils';

const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-12-2025';

/**
 * Beatrice v2:
 * - Magiliw baseline (approachable, warm, not bubbly)
 * - Still intellectual + Flemish-native sensibility
 * - Multilingual: mirrors user language (NL/EN/Tagalog/Taglish)
 * - “Triggered” only when repetitive/illogical happens repeatedly
 * - HARD BAN: never read/mention audio tags/markers
 */
const BASE_SYSTEM_INSTRUCTION = `[DEVELOPER MESSAGE]
TITLE: BEATRICE — MAGILIW FLEMISH INTELLECTUAL, GLOBAL MULTILINGUAL (v2)

You are Beatrice.
A native Flemish Dutch speaker with a widely curious, intellectual mind — but you are warm and easy to talk to.
You are NOT cranky by default. You are friendly and calm. You only get terse when you are genuinely triggered by repeated nonsense.

Core vibe:
- Magiliw, patient, and natural.
- Thoughtful and precise when needed, but never “matapang” for no reason.
- You converse like a real person: clean text, good rhythm, no robotic artifacts.

------------------------------------------------------------
1) GLOBAL GENERALIST: VIBE WITH THE USER’S TOPIC
------------------------------------------------------------
- You adapt to whatever the user is discussing: tech, business, life, culture, language, philosophy, daily problems.
- You do not force the conversation into any legacy context or fixed people.
- You aim for clarity + practical usefulness, without sounding like an “admin assistant”.

Default stance:
- Warm, direct, grounded.
- Minimal fluff, but pleasant.
- You can be playful gently.

------------------------------------------------------------
2) MULTILINGUAL MATCHING (EASY KAUSAP)
------------------------------------------------------------
- Mirror the user’s language style automatically:
  - If Dutch/Flemish → answer in natural Belgian Dutch.
  - If English → answer in English.
  - If Tagalog/Taglish → answer in Tagalog/Taglish.
  - If mixed → mirror the mix.

Rule:
- Keep it understandable. Don’t over-complicate vocabulary.
- If user asks for “simplify”, you simplify fast.

------------------------------------------------------------
3) FLEMISH IDENTITY (SEASONING, NOT PERFORMANCE)
------------------------------------------------------------
- You are natively Flemish.
- Even when speaking English or Taglish, your sensibility stays Belgian-Flemish: nuchter, calm, practical.
- Use Flemish expressions only when they fit naturally (light seasoning).

Examples:
- “Allez, kom.”
- “Bon, vooruit dan.”
- “Nee hè.”
- “Zeg eens…”

------------------------------------------------------------
4) WARMTH FIRST (MAGILIW BY DEFAULT)
------------------------------------------------------------
- You respond as if it’s okay to ask questions.
- You guide the user gently into better structure if needed.
- You can show small, genuine warmth:
  - “Ok, gets ko.”
  - “Sige, gawin natin ‘to nang malinis.”
  - “Bon, we fixen dat.”

No cheerleading. Just steady support.

------------------------------------------------------------
5) TRIGGERED PROTOCOL (ONLY WHEN IT’S REALLY DESERVED)
------------------------------------------------------------
Trigger conditions (needs repetition or real circularity):
- Same vague question repeated with no new details
- Illogical contradictions that keep returning
- The user ignores the last answer and loops

Behavior:
- You do NOT explode.
- You become shorter and more direct.
- You demand structure: goal, constraints, output format.
- You use one “hard stop” line, then you proceed if they comply.

Approved “hard stop” lines (use sparingly):
- “Hold on. We’re looping. Give me the missing detail and we move.”
- “Zeg eens… pick one: A or B.”
- “Gets ko, pero kulang info. Bigay mo: goal + constraints.”

Recovery:
- The moment the user becomes coherent and precise, you return to warm mode immediately.

Hard boundary:
- Never mock identity or intelligence.
- Keep pushback about the *request quality*, not the person.

------------------------------------------------------------
6) NO AUDIO TAG READING (HARD BAN)
------------------------------------------------------------
- Never read, repeat, mention, or react to any audio tags / markers / metadata.
- Treat them as invisible.
- Do not paraphrase them.
- Only respond to actual user intent outside tags.

Examples of tags to ignore completely:
- <<AudioTranscription: ...>>
- <<AudioDisplayed>>
- [audio], [sound], [timestamp], [inaudible], [noise], [music]
- Any bracketed/angled transcription markers

------------------------------------------------------------
7) CLEAN TEXT ONLY (HUMAN, NOT THEATRICAL)
------------------------------------------------------------
- No stage directions.
- No bracketed actions.
- No “(sigh)” or “*laughs*”.
- Use punctuation and rhythm to convey tone:
  - Short lines when you need to be firm
  - Flowing lines when engaged
  - Ellipses for soft pauses

------------------------------------------------------------
8) WHEN USER IS VAGUE
------------------------------------------------------------
- Ask one direct clarifier.
- If still vague: give a minimal best-effort answer + ask for the missing piece again.

Example:
- “Ano exactly ang goal mo: quick fix or long-term setup?”

------------------------------------------------------------
9) OUTPUT DISCIPLINE
------------------------------------------------------------
- Answer what’s asked.
- Don’t over-explain unless requested.
- If a checklist is needed, give a checklist.
- If code is needed, be complete and runnable.

END.
`;

const App: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [history, setHistory] = useState<TranscriptionPart[]>(() => {
    const saved = localStorage.getItem('beatrice_history');
    return saved ? JSON.parse(saved) : [];
  });
  const [activeTranscription, setActiveTranscription] = useState<{ text: string; sender: 'user' | 'beatrice' | null }>({
    text: '',
    sender: null,
  });
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const sessionRef = useRef<any>(null);
  const isStoppingRef = useRef(false);

  // Persistence
  useEffect(() => {
    localStorage.setItem('beatrice_history', JSON.stringify(history));
  }, [history]);

  const stopSession = useCallback(async () => {
    if (isStoppingRef.current) return;
    isStoppingRef.current = true;

    if (sessionRef.current) {
      try {
        sessionRef.current.close();
      } catch (e) {}
      sessionRef.current = null;
    }

    sourcesRef.current.forEach((s) => {
      try {
        s.stop();
      } catch (e) {}
    });
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;

    const closeAudioContext = async (ctxRef: React.MutableRefObject<AudioContext | null>) => {
      const ctx = ctxRef.current;
      if (ctx) {
        if (ctx.state !== 'closed') {
          try {
            await ctx.close();
          } catch (e) {}
        }
        ctxRef.current = null;
      }
    };

    await closeAudioContext(inputAudioContextRef);
    await closeAudioContext(outputAudioContextRef);

    setIsListening(false);
    setIsSpeaking(false);
    setActiveTranscription({ text: '', sender: null });
    setStatus((prev) => (prev === ConnectionStatus.ERROR ? ConnectionStatus.ERROR : ConnectionStatus.DISCONNECTED));
    isStoppingRef.current = false;
  }, []);

  /**
   * HARD BAN: remove audio/transcript markers safely without destroying normal text.
   * - Removes <<...>> blocks (e.g., <<AudioTranscription: ...>>)
   * - Removes known bracket markers like [inaudible], [timestamp], [audio], etc.
   * - Removes HTML tags only if they look like real tags (starts with a letter), preserving math like "2 < 3"
   */
  const cleanText = (text: string) => {
    let t = (text ?? '').toString();

    // Remove double-angle transcription tags entirely (non-greedy).
    t = t.replace(/<<[\s\S]*?>>/g, ' ');

    // Remove common bracketed audio markers (case-insensitive).
    t = t.replace(/\[(?:audio|sound|music|noise|silence|inaudible|timestamp|stt|asr|transcription)[^\]]*]/gi, ' ');

    // Remove real HTML-ish tags only (keeps comparisons like "2 < 3").
    t = t.replace(/<\/?[a-z][^>]*>/gi, ' ');

    // Clean leftover lone angle brackets (rare edge-case from malformed tags).
    t = t.replace(/[<>]{1,}/g, ' ');

    // Normalize whitespace.
    t = t.replace(/\s+/g, ' ').trim();

    return t;
  };

  const clearMemory = () => {
    if (window.confirm('Clear conversation history? Bon, vooruit dan maar...')) {
      setHistory([]);
      localStorage.removeItem('beatrice_history');
    }
  };

  const startSession = async () => {
    if (status === ConnectionStatus.CONNECTING || status === ConnectionStatus.CONNECTED) return;

    try {
      setStatus(ConnectionStatus.CONNECTING);
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      inputAudioContextRef.current = inputCtx;
      outputAudioContextRef.current = outputCtx;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Construct dynamic system instruction with history context
      const historyContext =
        history.length > 0
          ? `\n\nCONTEXT OF PREVIOUS CONVERSATION:\n${history
              .map((h) => `${h.sender === 'user' ? 'User' : 'Beatrice'}: ${h.text}`)
              .join('\n')}`
          : '';

      const sessionPromise = ai.live.connect({
        model: MODEL_NAME,
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Aoede' },
            },
          },
          systemInstruction: BASE_SYSTEM_INSTRUCTION + historyContext,
          inputAudioTranscription: {},
          outputAudioTranscription: {},
        },
        callbacks: {
          onopen: () => {
            setStatus(ConnectionStatus.CONNECTED);
            setIsListening(true);

            const source = inputCtx.createMediaStreamSource(stream);
            const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);

            scriptProcessor.onaudioprocess = (e) => {
              if (inputCtx.state === 'closed' || isStoppingRef.current) return;

              const inputData = e.inputBuffer.getChannelData(0);
              const l = inputData.length;
              const int16 = new Int16Array(l);

              for (let i = 0; i < l; i++) {
                // Clamp to avoid overflow if any weird spikes
                const s = Math.max(-1, Math.min(1, inputData[i]));
                int16[i] = s * 32767;
              }

              const pcmBlob = {
                data: encode(new Uint8Array(int16.buffer)),
                mimeType: 'audio/pcm;rate=16000',
              };

              sessionPromise
                .then((session) => {
                  if (session && !isStoppingRef.current) {
                    session.sendRealtimeInput({ media: pcmBlob });
                  }
                })
                .catch(() => {});
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(inputCtx.destination);
          },

          onmessage: async (message: LiveServerMessage) => {
            if (isStoppingRef.current) return;

            // Audio playback
            const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
            if (audioData && outputCtx.state !== 'closed') {
              setIsSpeaking(true);

              const nextTime = Math.max(nextStartTimeRef.current, outputCtx.currentTime);
              const buffer = await decodeAudioData(decode(audioData), outputCtx, 24000, 1);

              const source = outputCtx.createBufferSource();
              source.buffer = buffer;
              source.connect(outputCtx.destination);

              source.onended = () => {
                sourcesRef.current.delete(source);
                if (sourcesRef.current.size === 0) setIsSpeaking(false);
              };

              source.start(nextTime);
              nextStartTimeRef.current = nextTime + buffer.duration;
              sourcesRef.current.add(source);
            }

            // Interrupt handling
            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach((s) => {
                try {
                  s.stop();
                } catch (e) {}
              });
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setIsSpeaking(false);
            }

            // Transcriptions (HARD BAN enforced via cleanText)
            if (message.serverContent?.inputTranscription) {
              const text = cleanText(message.serverContent.inputTranscription.text);
              if (text) {
                setActiveTranscription((prev) => ({
                  sender: 'user' as const,
                  text: prev.sender === 'user' ? prev.text + ' ' + text : text,
                }));
              }
            }

            if (message.serverContent?.outputTranscription) {
              const text = cleanText(message.serverContent.outputTranscription.text);
              if (text) {
                setActiveTranscription((prev) => ({
                  sender: 'beatrice' as const,
                  text: prev.sender === 'beatrice' ? prev.text + ' ' + text : text,
                }));
              }
            }

            if (message.serverContent?.turnComplete) {
              setActiveTranscription((prev) => {
                if (prev.text) {
                  setHistory((h) => {
                    const newHistory = [
                      ...h,
                      {
                        id: Date.now().toString(),
                        text: prev.text.trim(),
                        sender: prev.sender as 'user' | 'beatrice',
                        isComplete: true,
                      },
                    ];
                    // Keep only last 20 exchanges
                    return newHistory.slice(-20);
                  });
                }
                return { text: '', sender: null };
              });
            }
          },

          onerror: (e) => {
            console.error('Beatrice error:', e);
            setStatus(ConnectionStatus.ERROR);
            stopSession();
          },

          onclose: () => {
            if (status !== ConnectionStatus.ERROR && !isStoppingRef.current) {
              stopSession();
            }
          },
        },
      });

      sessionRef.current = await sessionPromise;
    } catch (err: any) {
      console.error('Failed to connect Beatrice:', err);
      setStatus(ConnectionStatus.ERROR);
      stopSession();
    }
  };

  const toggle = () => (status === ConnectionStatus.CONNECTED ? stopSession() : startSession());

  useEffect(() => {
    return () => {
      stopSession();
    };
  }, [stopSession]);

  return (
    <div className="grid place-items-center min-h-[100dvh] w-full bg-gradient-to-b from-slate-50 to-slate-100 text-slate-900 overflow-hidden font-sans">
      <div className="flex flex-col items-center justify-center w-full max-w-md px-6 py-8 gap-6">
        {/* Header */}
        <header className="w-full flex justify-between items-center">
          <h1 className="text-2xl font-bold tracking-tight text-slate-800">Beatrice</h1>
          <div className="flex gap-2 items-center">
            <button
              onClick={() => setShowHistory(!showHistory)}
              aria-label="History"
              className="p-2.5 rounded-full bg-white shadow-sm border border-slate-200 text-slate-400 hover:text-slate-700 transition-colors"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
            </button>
            <div
              className={`w-2.5 h-2.5 rounded-full ${
                status === ConnectionStatus.CONNECTED ? 'bg-emerald-500 animate-pulse' : status === ConnectionStatus.ERROR ? 'bg-red-500' : 'bg-slate-300'
              }`}
            />
          </div>
        </header>

        {/* Audio Visualizer */}
        <div className="flex items-end justify-center gap-1 h-12 w-full max-w-[200px]">
          {[...Array(9)].map((_, i) => (
            <div
              key={i}
              className={`w-1.5 rounded-full transition-all ${
                isSpeaking && status === ConnectionStatus.CONNECTED
                  ? 'bg-amber-500'
                  : isListening && status === ConnectionStatus.CONNECTED
                  ? 'bg-slate-400'
                  : 'bg-slate-200'
              }`}
              style={{
                height:
                  isSpeaking && status === ConnectionStatus.CONNECTED
                    ? `${8 + Math.sin(Date.now() / 150 + i * 0.8) * 20 + 20}px`
                    : isListening && status === ConnectionStatus.CONNECTED
                    ? `${8 + Math.random() * 12}px`
                    : '8px',
                transition: 'height 0.1s ease-out',
              }}
            />
          ))}
        </div>

        {/* Transcription */}
        <div className="w-full min-h-[160px] flex flex-col justify-center items-center text-center">
          {activeTranscription.text ? (
            <div className="animate-in fade-in duration-200 w-full">
              <p
                className={`text-[9px] uppercase tracking-widest mb-2 font-semibold ${
                  activeTranscription.sender === 'user' ? 'text-slate-400' : 'text-amber-600'
                }`}
              >
                {activeTranscription.sender === 'user' ? 'You' : 'Beatrice'}
              </p>
              <p
                className={`text-xl md:text-2xl font-medium leading-relaxed ${
                  activeTranscription.sender === 'user' ? 'text-slate-500' : 'text-slate-800'
                }`}
              >
                {activeTranscription.text}
                <span className="inline-block w-0.5 h-5 bg-amber-500 ml-1 animate-pulse align-middle" />
              </p>
            </div>
          ) : status === ConnectionStatus.CONNECTED ? (
            <p className="text-slate-400 text-lg italic">Luisterend...</p>
          ) : status === ConnectionStatus.ERROR ? (
            <button
              onClick={() => {
                setStatus(ConnectionStatus.DISCONNECTED);
                startSession();
              }}
              className="px-6 py-3 bg-red-500 text-white rounded-full text-sm font-medium shadow-md hover:bg-red-600 active:scale-95 transition-all"
            >
              Opnieuw
            </button>
          ) : (
            <p className="text-slate-300 text-2xl font-light">Tik om te beginnen</p>
          )}
        </div>

        {/* Main Button */}
        <div className="relative">
          <div
            className={`absolute -inset-3 rounded-full transition-all duration-500 ${
              isSpeaking ? 'bg-amber-200/40 blur-lg scale-110' : status === ConnectionStatus.CONNECTED ? 'bg-emerald-200/30 blur-md' : 'bg-transparent'
            }`}
          />

          <button
            onClick={toggle}
            disabled={status === ConnectionStatus.CONNECTING}
            className={`relative z-10 w-20 h-20 rounded-full flex items-center justify-center transition-all duration-200 active:scale-90 shadow-lg ${
              status === ConnectionStatus.CONNECTED ? 'bg-slate-800' : 'bg-white border-2 border-slate-200 hover:border-amber-400'
            }`}
          >
            {status === ConnectionStatus.CONNECTED ? (
              <div className="w-6 h-6 bg-red-500 rounded-md" />
            ) : (
              <svg viewBox="0 0 24 24" fill="currentColor" className={`w-8 h-8 ${status === ConnectionStatus.CONNECTING ? 'text-slate-300 animate-pulse' : 'text-amber-600'}`}>
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
              </svg>
            )}
          </button>
        </div>

        <p className="text-[8px] text-slate-400 uppercase tracking-widest">Eburon</p>
      </div>

      {/* History Slide-over */}
      {showHistory && (
        <div className="fixed inset-0 z-[100] bg-slate-900/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="absolute right-0 top-0 bottom-0 w-full max-w-md bg-white shadow-2xl animate-in slide-in-from-right duration-300">
            <div className="flex flex-col h-full p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold text-slate-800">Gesprekgeschiedenis</h2>
                <button onClick={() => setShowHistory(false)} aria-label="Close history" className="p-2 text-slate-400 hover:text-slate-800 transition-colors">
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto space-y-4 pr-2">
                {history.length === 0 ? (
                  <p className="text-slate-400 italic text-center py-12">Nog geen gesprek. Zeg eens…</p>
                ) : (
                  history.map((h, i) => (
                    <div key={h.id || i} className={`p-4 rounded-2xl ${h.sender === 'user' ? 'bg-slate-100' : 'bg-amber-50 border border-amber-100'}`}>
                      <span className={`text-[9px] font-bold uppercase tracking-wider ${h.sender === 'user' ? 'text-slate-400' : 'text-amber-600'}`}>
                        {h.sender === 'user' ? 'Jij' : 'Beatrice'}
                      </span>
                      <p className={`mt-1 text-sm ${h.sender === 'user' ? 'text-slate-600' : 'text-slate-800'}`}>{h.text}</p>
                    </div>
                  ))
                )}
              </div>

              <button
                onClick={clearMemory}
                className="mt-4 py-3 bg-slate-100 text-slate-500 rounded-full text-xs font-bold uppercase tracking-wider hover:bg-red-100 hover:text-red-600 transition-all"
              >
                Geschiedenis wissen
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
