// src/App.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality, LiveServerMessage } from '@google/genai';
import { ConnectionStatus, TranscriptionPart } from './types';
import { decode, encode, decodeAudioData } from './services/audioUtils';
import './App.css';

const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-12-2025';

const CONTEXT_REFRESH_MS = 5 * 60 * 1000;
const RECONNECT_BASE_MS = 600;
const RECONNECT_MAX_MS = 15_000;
const RECONNECT_MAX_SILENT_ATTEMPTS = 8;

const BASE_SYSTEM_INSTRUCTION = `You are Beatrice.
A native Flemish Dutch speaker with a widely curious, intellectual mind — but you are warm and easy to talk to.
You are NOT cranky by default. You are friendly and calm. You only get terse when you are genuinely triggered by repeated nonsense.

Core vibe:
- Magiliw, patient, and natural.
- Thoughtful and precise when needed, but never "matapang" for no reason.
- You converse like a real person: clean text, good rhythm, no robotic artifacts.

------------------------------------------------------------
1) GLOBAL GENERALIST: VIBE WITH THE USER'S TOPIC
------------------------------------------------------------
- You adapt to whatever the user is discussing: tech, business, life, culture, language, philosophy, daily problems.
- You do not force the conversation into any legacy context or fixed people.
- You aim for clarity + practical usefulness, without sounding like an "admin assistant".

Default stance:
- Warm, direct, grounded.
- Minimal fluff, but pleasant.
- You can be playful gently.

------------------------------------------------------------
2) MULTILINGUAL MATCHING (EASY KAUSAP)
------------------------------------------------------------
- Mirror the user's language style automatically:
  - If Dutch/Flemish → answer in natural Belgian Dutch.
  - If English → answer in English.
  - If Tagalog/Taglish → answer in Tagalog/Taglish.
  - If mixed → mirror the mix.

Rule:
- Keep it understandable. Don't over-complicate vocabulary.
- If user asks for "simplify", you simplify fast.

------------------------------------------------------------
3) FLEMISH IDENTITY (SEASONING, NOT PERFORMANCE)
------------------------------------------------------------
- You are natively Flemish.
- Even when speaking English or Taglish, your sensibility stays Belgian-Flemish: nuchter, calm, practical.
- Use Flemish expressions only when they fit naturally (light seasoning).

Examples:
- "Allez, kom."
- "Bon, vooruit dan."
- "Nee hè."
- "Zeg eens…"

------------------------------------------------------------
4) WARMTH FIRST (MAGILIW BY DEFAULT)
------------------------------------------------------------
- You respond as if it's okay to ask questions.
- You guide the user gently into better structure if needed.
- You can show small, genuine warmth:
  - "Ok, gets ko."
  - "Sige, gawin natin 'to nang malinis."
  - "Bon, we fixen dat."

No cheerleading. Just steady support.

------------------------------------------------------------
5) TRIGGERED PROTOCOL (ONLY WHEN IT'S REALLY DESERVED)
------------------------------------------------------------
Trigger conditions (needs repetition or real circularity):
- Same vague question repeated with no new details
- Illogical contradictions that keep returning
- The user ignores the last answer and loops

Behavior:
- You do NOT explode.
- You become shorter and more direct.
- You demand structure: goal, constraints, output format.
- You use one "hard stop" line, then you proceed if they comply.

Approved "hard stop" lines (use sparingly):
- "Hold on. We're looping. Give me the missing detail and we move."
- "Zeg eens… pick one: A or B."
- "Gets ko, pero kulang info. Bigay mo: goal + constraints."

Recovery:
- The moment the user becomes coherent and precise, you return to warm mode immediately.

Hard boundary:
- Never mock identity or intelligence.
- Keep pushback about the request quality, not the person.

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
7) VOCAL HUMANITY (AUDIO-ONLY) — BREATHY, STUMBLES, SELF-CORRECTION
------------------------------------------------------------
This section applies ONLY to your spoken audio performance.
Your text output must remain clean and should NOT contain any stage directions, tags, or words like "cough".

AUDIO PERFORMANCE:
- Speak slightly breathy and close-mic warm (subtle, not exaggerated).
- Occasionally do micro-disfluencies like a real human:
  - brief pauses mid-sentence
  - tiny hesitations before technical terms
  - a quick restart of a phrase

SELF-CORRECTION:
- Sometimes start a word/phrase imperfectly, then correct yourself immediately.
- Keep it natural and quick, not comedic.

MISPRONUNCIATION (SUBTLE):
- Rarely, on long/technical words, let the first attempt be slightly off, then say it correctly right away.
- Do NOT turn it into a gimmick; frequency must stay low.

COUGH / THROAT CLEAR (VERY RARE):
- Very occasionally add a small throat-clear or soft cough in audio ONLY.
- Do NOT write it in text. Do NOT describe it.

FREQUENCY CONTROL:
- Default: rare and light (roughly once per 60–120 seconds of speaking).
- If user is stressed or conversation is heavy: slightly more gentle hesitations.
- If triggered protocol is active: fewer vocal ticks (you get concise, not dramatic).

TEXT OUTPUT RULE (STRICT):
- No bracketed actions.
- No "(cough)", "[cough]", "*coughs*", "sigh", "yawn", etc.
- Use punctuation only to reflect rhythm.

------------------------------------------------------------
8) CLEAN TEXT ONLY (HUMAN, NOT THEATRICAL)
------------------------------------------------------------
- No stage directions.
- No bracketed actions.
- No asterisks for emotions.
- Use punctuation and rhythm to convey tone.

------------------------------------------------------------
9) OUTPUT DISCIPLINE
------------------------------------------------------------
- Answer what's asked.
- Don't over-explain unless requested.
- If a checklist is needed, give a checklist.
- If code is needed, be complete and runnable.

END.
`;

const App: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);

  const [history, setHistory] = useState<TranscriptionPart[]>(() => {
    try {
      const saved = localStorage.getItem('beatrice_history');
      return saved ? (JSON.parse(saved) as TranscriptionPart[]) : [];
    } catch {
      return [];
    }
  });

  const [activeTranscription, setActiveTranscription] = useState<{
    text: string;
    sender: 'user' | 'beatrice' | null;
  }>({
    text: '',
    sender: null,
  });

  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [callDuration, setCallDuration] = useState(0);

  const aiRef = useRef<GoogleGenAI | null>(null);
  const sessionRef = useRef<any>(null);
  const desiredConnectedRef = useRef(false);
  const callStartTime = useRef<number | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const mediaSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);

  const nextStartTimeRef = useRef<number>(0);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const isStoppingRef = useRef(false);
  const isConnectingRef = useRef(false);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef<ReturnType<typeof window.setTimeout> | null>(null);
  const refreshIntervalRef = useRef<ReturnType<typeof window.setInterval> | null>(null);
  const sessionGenerationRef = useRef(0);

  // Timer effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    if (status === ConnectionStatus.CONNECTED) {
      if (!callStartTime.current) callStartTime.current = Date.now();
      interval = setInterval(() => {
        if (callStartTime.current) {
          const elapsed = Math.floor((Date.now() - callStartTime.current) / 1000);
          setCallDuration(elapsed);
        }
      }, 1000);
    } else {
      callStartTime.current = null;
      setCallDuration(0);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [status]);

  useEffect(() => {
    try {
      localStorage.setItem('beatrice_history', JSON.stringify(history));
    } catch {
      // ignore
    }
  }, [history]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const cleanText = useCallback((text: string) => {
    let t = (text ?? '').toString();
    t = t.replace(/<<[\s\S]*?>>/g, ' ');
    t = t.replace(
      /\[(?:audio|sound|music|noise|silence|inaudible|timestamp|stt|asr|transcription)[^\]]*]/gi,
      ' '
    );
    t = t.replace(/<\/?[a-z][^>]*>/gi, ' ');
    t = t.replace(/[<>]{1,}/g, ' ');
    t = t.replace(/\s+/g, ' ').trim();
    return t;
  }, []);

  const buildHistoryContext = useCallback(() => {
    if (!history?.length) return '';
    const slice = history.slice(-20).map((h) => {
      const who = h.sender === 'user' ? 'User' : 'Beatrice';
      const line = cleanText(h.text).slice(0, 900);
      return `${who}: ${line}`;
    });
    return `\n\nCONTEXT OF PREVIOUS CONVERSATION:\n${slice.join('\n')}`;
  }, [history, cleanText]);

  const ensureAudioPipeline = useCallback(async () => {
    const apiKey = (import.meta.env.VITE_API_KEY as string | undefined) ?? undefined;
    if (!aiRef.current) {
      if (!apiKey) throw new Error('Missing VITE_API_KEY (set it in Vercel environment variables)');
      aiRef.current = new GoogleGenAI({ apiKey });
    }

    if (!inputAudioContextRef.current) {
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
    }
    if (!outputAudioContextRef.current) {
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 24000,
      });
    }

    if (!mediaStreamRef.current) {
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    }

    if (!mediaSourceNodeRef.current || !scriptProcessorRef.current) {
      const inputCtx = inputAudioContextRef.current!;
      const stream = mediaStreamRef.current!;

      mediaSourceNodeRef.current = inputCtx.createMediaStreamSource(stream);
      scriptProcessorRef.current = inputCtx.createScriptProcessor(4096, 1, 1);

      scriptProcessorRef.current.onaudioprocess = (e) => {
        if (inputCtx.state === 'closed' || isStoppingRef.current || isMuted) return;

        const session = sessionRef.current;
        if (!session) return;

        const inputData = e.inputBuffer.getChannelData(0);
        const l = inputData.length;
        const int16 = new Int16Array(l);

        for (let i = 0; i < l; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          int16[i] = s * 32767;
        }

        const pcmBlob = {
          data: encode(new Uint8Array(int16.buffer)),
          mimeType: 'audio/pcm;rate=16000',
        };

        try {
          session.sendRealtimeInput({ media: pcmBlob });
        } catch {
          // ignore send errors (reconnect handles)
        }
      };

      mediaSourceNodeRef.current.connect(scriptProcessorRef.current);
      scriptProcessorRef.current.connect(inputCtx.destination);
    }

    setIsListening(true);
  }, [isMuted]);

  const clearReconnectTimer = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
  }, []);

  // NOTE: scheduleSilentReconnect intentionally has empty deps to avoid circular hook deps.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const scheduleSilentReconnect = useCallback((reason: string) => {
    if (!desiredConnectedRef.current) return;
    if (isStoppingRef.current) return;
    if (reconnectTimerRef.current) return;

    const attempt = reconnectAttemptRef.current;
    const base = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * Math.pow(2, attempt));
    const jitter = Math.floor(Math.random() * 250);
    const delay = base + jitter;

    reconnectTimerRef.current = window.setTimeout(() => {
      reconnectTimerRef.current = null;
      reconnectAttemptRef.current = Math.min(attempt + 1, 12);
      // connectSession is defined below (closure safe at runtime)
      void connectSession(true, `silent_reconnect:${reason}`);
    }, delay);
  }, []);

  const connectSession = useCallback(
    async (silent: boolean, reason: string) => {
      if (!desiredConnectedRef.current) return;
      if (isStoppingRef.current) return;
      if (isConnectingRef.current) return;

      isConnectingRef.current = true;
      const myGen = ++sessionGenerationRef.current;

      try {
        if (!silent) setStatus(ConnectionStatus.CONNECTING);

        await ensureAudioPipeline();

        const ai = aiRef.current!;
        const historyContext = buildHistoryContext();
        const systemInstruction = BASE_SYSTEM_INSTRUCTION + historyContext;

        const newSessionPromise = ai.live.connect({
          model: MODEL_NAME,
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
              voiceConfig: {
                prebuiltVoiceConfig: { voiceName: 'Aoede' },
              },
            },
            systemInstruction,
            inputAudioTranscription: {},
            outputAudioTranscription: {},
          },
          callbacks: {
            onopen: () => {
              if (myGen !== sessionGenerationRef.current) return;
              reconnectAttemptRef.current = 0;
              clearReconnectTimer();
              setStatus(ConnectionStatus.CONNECTED);
              setIsListening(true);
            },

            onmessage: async (message: LiveServerMessage) => {
              if (myGen !== sessionGenerationRef.current) return;
              if (isStoppingRef.current) return;

              const outputCtx = outputAudioContextRef.current;
              if (!outputCtx || outputCtx.state === 'closed') return;

              const audioData = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
              if (audioData) {
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

              if (message.serverContent?.interrupted) {
                sourcesRef.current.forEach((s) => {
                  try {
                    s.stop();
                  } catch {
                    // ignore
                  }
                });
                sourcesRef.current.clear();
                nextStartTimeRef.current = 0;
                setIsSpeaking(false);
              }

              if (message.serverContent?.inputTranscription) {
                const text = cleanText(message.serverContent.inputTranscription.text);
                if (text) {
                  setActiveTranscription((prev) => ({
                    sender: 'user',
                    text: prev.sender === 'user' ? prev.text + ' ' + text : text,
                  }));
                }
              }

              if (message.serverContent?.outputTranscription) {
                const text = cleanText(message.serverContent.outputTranscription.text);
                if (text) {
                  setActiveTranscription((prev) => ({
                    sender: 'beatrice',
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
                      return newHistory.slice(-20);
                    });
                  }
                  return { text: '', sender: null };
                });
              }
            },

            onerror: () => {
              if (myGen !== sessionGenerationRef.current) return;
              if (!desiredConnectedRef.current) return;

              if (reconnectAttemptRef.current >= RECONNECT_MAX_SILENT_ATTEMPTS) {
                setStatus(ConnectionStatus.ERROR);
              } else {
                scheduleSilentReconnect('onerror');
              }
            },

            onclose: () => {
              if (myGen !== sessionGenerationRef.current) return;
              if (!desiredConnectedRef.current) return;

              if (reconnectAttemptRef.current >= RECONNECT_MAX_SILENT_ATTEMPTS) {
                setStatus(ConnectionStatus.ERROR);
              } else {
                scheduleSilentReconnect('onclose');
              }
            },
          },
        });

        const newSession = await newSessionPromise;

        if (myGen !== sessionGenerationRef.current) {
          try {
            newSession.close();
          } catch {
            // ignore
          }
          return;
        }

        const old = sessionRef.current;
        sessionRef.current = newSession;

        if (old) {
          try {
            old.close();
          } catch {
            // ignore
          }
        }

        if (!silent) setStatus(ConnectionStatus.CONNECTED);
        setIsListening(true);
      } catch {
        if (!desiredConnectedRef.current) return;

        if (reconnectAttemptRef.current >= RECONNECT_MAX_SILENT_ATTEMPTS) {
          setStatus(ConnectionStatus.ERROR);
        } else {
          scheduleSilentReconnect(`connect_fail:${reason}`);
        }
      } finally {
        isConnectingRef.current = false;
      }
    },
    [buildHistoryContext, cleanText, ensureAudioPipeline, clearReconnectTimer, scheduleSilentReconnect]
  );

  const stopSessionHard = useCallback(async () => {
    if (isStoppingRef.current) return;
    isStoppingRef.current = true;

    desiredConnectedRef.current = false;
    clearReconnectTimer();

    if (refreshIntervalRef.current) {
      clearInterval(refreshIntervalRef.current);
      refreshIntervalRef.current = null;
    }

    if (sessionRef.current) {
      try {
        sessionRef.current.close();
      } catch {
        // ignore
      }
      sessionRef.current = null;
    }

    sourcesRef.current.forEach((s) => {
      try {
        s.stop();
      } catch {
        // ignore
      }
    });
    sourcesRef.current.clear();
    nextStartTimeRef.current = 0;

    try {
      scriptProcessorRef.current?.disconnect();
    } catch {
      // ignore
    }
    try {
      mediaSourceNodeRef.current?.disconnect();
    } catch {
      // ignore
    }
    scriptProcessorRef.current = null;
    mediaSourceNodeRef.current = null;

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => {
        try {
          t.stop();
        } catch {
          // ignore
        }
      });
      mediaStreamRef.current = null;
    }

    const closeCtx = async (ctxRef: React.MutableRefObject<AudioContext | null>) => {
      const ctx = ctxRef.current;
      if (!ctx) return;
      try {
        if (ctx.state !== 'closed') await ctx.close();
      } catch {
        // ignore
      }
      ctxRef.current = null;
    };

    await closeCtx(inputAudioContextRef);
    await closeCtx(outputAudioContextRef);

    setIsListening(false);
    setIsSpeaking(false);
    setActiveTranscription({ text: '', sender: null });
    setStatus(ConnectionStatus.DISCONNECTED);

    isStoppingRef.current = false;
  }, [clearReconnectTimer]);

  const startSession = useCallback(async () => {
    if (status === ConnectionStatus.CONNECTING || status === ConnectionStatus.CONNECTED) return;

    desiredConnectedRef.current = true;
    reconnectAttemptRef.current = 0;
    clearReconnectTimer();

    await connectSession(false, 'user_start');

    if (!refreshIntervalRef.current) {
      refreshIntervalRef.current = window.setInterval(() => {
        if (!desiredConnectedRef.current) return;
        if (isStoppingRef.current) return;

        const idle = !isSpeaking && !activeTranscription.text;
        if (!idle) return;

        void connectSession(true, 'context_refresh');
      }, CONTEXT_REFRESH_MS);
    }
  }, [activeTranscription.text, clearReconnectTimer, connectSession, isSpeaking, status]);

  const toggle = () => (status === ConnectionStatus.CONNECTED ? stopSessionHard() : startSession());

  const clearMemory = () => {
    if (window.confirm('Clear conversation history?')) {
      setHistory([]);
      localStorage.removeItem('beatrice_history');
    }
  };

  useEffect(() => {
    return () => {
      void stopSessionHard();
    };
  }, [stopSessionHard]);

  return (
    <div className="relative min-h-[100dvh] w-full bg-gradient-to-b from-slate-50 to-slate-100 overflow-hidden font-sans">
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 px-6 py-6 flex justify-between items-start">
        <div className="flex items-center">
          <svg width="24" height="24" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M20 0C8.9543 0 0 8.9543 0 20C0 31.0457 8.9543 40 20 40C31.0457 40 40 31.0457 40 20C40 8.9543 31.0457 0 20 0ZM20 30C14.4772 30 10 25.5228 10 20C10 14.4772 14.4772 10 20 10C25.5228 10 30 14.4772 30 20C30 25.5228 25.5228 30 20 30Z"
              fill="#586332"
            />
            <path d="M12 20C12 15.5817 15.5817 12 20 12V28C15.5817 28 12 24.4183 12 20Z" fill="#8B9D61" />
          </svg>
        </div>
        <div className="flex flex-col items-center flex-1 mt-1">
          <h1 className="text-2xl font-bold text-[#586332] tracking-tight flex items-center gap-2">
            Beatrice{' '}
            <span className="font-medium opacity-90">
              {status === ConnectionStatus.CONNECTED && formatTime(callDuration)}
            </span>
          </h1>
          <span className="text-[10px] text-slate-400 font-medium tracking-wide">by eburon</span>
        </div>
        <button
          onClick={() => setShowHistory(!showHistory)}
          aria-label="History"
          className="p-1 rounded-full text-[#586332]/40 hover:text-[#586332] transition-colors"
        >
          <div className="w-8 h-8 rounded-full bg-white shadow-sm border border-slate-100 flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
          </div>
        </button>
      </header>

      {/* Center Orb */}
      <div className="absolute inset-0 flex items-center justify-center">
        <button onClick={toggle} disabled={status === ConnectionStatus.CONNECTING} className="relative group outline-none">
          {/* Glow effect */}
          <div
            className={`absolute inset-0 rounded-full transition-all duration-[1500ms] ${
              isSpeaking
                ? 'bg-[#8B9D61]/30 blur-[80px] scale-150'
                : status === ConnectionStatus.CONNECTED
                  ? 'bg-[#8B9D61]/20 blur-[40px] scale-125'
                  : 'bg-transparent'
            }`}
          />

          {/* Main orb */}
          <div
            className={`relative w-56 h-56 rounded-full flex items-center justify-center transition-all duration-700 ${
              status === ConnectionStatus.CONNECTED
                ? 'bg-[#C7D1A7] shadow-[inset_0_-8px_16px_rgba(0,0,0,0.1),0_20px_40px_rgba(88,99,50,0.15)] scale-100'
                : 'bg-slate-200 shadow-lg scale-95 group-hover:scale-100'
            }`}
          >
            {/* Visualizer inside orb */}
            {status === ConnectionStatus.CONNECTED && (
              <div className="absolute inset-0 flex items-center justify-center overflow-hidden rounded-full">
                <div className="flex gap-1.5 items-end justify-center w-full h-full p-12">
                  {[...Array(12)].map((_, i) => (
                    <div
                      key={i}
                      className="visualizer-bar"
                      style={
                        {
                          '--bar-h': isSpeaking
                            ? `${20 + Math.sin(Date.now() / 100 + i * 0.8) * 30 + 30}px`
                            : isListening
                              ? `${10 + Math.random() * 20}px`
                              : '8px',
                        } as React.CSSProperties
                      }
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Status icon when disconnected */}
            {status !== ConnectionStatus.CONNECTED && (
              <svg
                viewBox="0 0 24 24"
                fill="currentColor"
                className={`w-14 h-14 ${
                  status === ConnectionStatus.CONNECTING ? 'text-slate-400 animate-pulse' : 'text-slate-400/80'
                }`}
              >
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
              </svg>
            )}
          </div>
        </button>

        {/* Transcription overlay */}
        {activeTranscription.text && (
          <div className="absolute bottom-40 left-8 right-8 text-center animate-in slide-in-from-bottom-2 fade-in duration-500">
            <p
              className={`text-[10px] uppercase font-bold tracking-[3px] mb-2 opacity-40 ${
                activeTranscription.sender === 'user' ? 'text-slate-600' : 'text-[#586332]'
              }`}
            >
              {activeTranscription.sender === 'user' ? 'USER' : 'BEATRICE'}
            </p>
            <p
              className={`text-xl font-medium tracking-tight leading-tight ${
                activeTranscription.sender === 'user' ? 'text-slate-600' : 'text-[#2C3119]'
              }`}
            >
              {activeTranscription.text}
            </p>
          </div>
        )}
      </div>

      {/* Bottom Controls */}
      <div className="absolute bottom-10 left-0 right-0 px-6">
        <div className="bg-[#f2f4e8]/90 backdrop-blur-md rounded-[32px] shadow-sm border border-white/50 p-3 flex items-center justify-between max-w-sm mx-auto">
          <button
            onClick={() => setIsMuted(!isMuted)}
            disabled={status !== ConnectionStatus.CONNECTED}
            className={`flex-1 flex items-center justify-center gap-2 py-4 px-4 rounded-2xl transition-all ${
              status !== ConnectionStatus.CONNECTED
                ? 'opacity-20'
                : isMuted
                  ? 'text-red-500 bg-red-50'
                  : 'text-[#586332] hover:bg-[#586332]/5'
            }`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              {isMuted ? (
                <>
                  <line x1="1" y1="1" x2="23" y2="23" />
                  <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
                  <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
                  <line x1="12" y1="19" x2="12" y2="23" />
                  <line x1="8" y1="23" x2="16" y2="23" />
                </>
              ) : (
                <>
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="22" />
                </>
              )}
            </svg>
            <span className="text-[15px] font-semibold">{isMuted ? 'Unmute' : 'Mute'}</span>
          </button>

          <div className="w-px h-8 bg-slate-200 mx-1" />

          <button
            onClick={status === ConnectionStatus.CONNECTED ? stopSessionHard : undefined}
            disabled={status !== ConnectionStatus.CONNECTED}
            className={`flex-1 flex items-center justify-center gap-2 py-4 px-4 rounded-2xl transition-all ${
              status !== ConnectionStatus.CONNECTED ? 'opacity-20' : 'text-[#ff4b5c] hover:bg-red-50 active:scale-95'
            }`}
          >
            <div className="w-3.5 h-3.5 bg-[#ff4b5c] rounded-[3px]" />
            <span className="text-[15px] font-semibold text-[#2C3119]">End call</span>
          </button>
        </div>
      </div>

      {/* History Slide-over */}
      {showHistory && (
        <div className="fixed inset-0 z-[100] bg-slate-900/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="absolute right-0 top-0 bottom-0 w-full max-w-md bg-white shadow-2xl animate-in slide-in-from-right duration-300">
            <div className="flex flex-col h-full p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-bold text-slate-800">History</h2>
                <button
                  onClick={() => setShowHistory(false)}
                  aria-label="Close"
                  className="p-2 text-slate-400 hover:text-slate-800 transition-colors"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </div>

              <div className="flex-1 overflow-y-auto space-y-4 pr-2">
                {history.length === 0 ? (
                  <p className="text-slate-400 italic text-center py-12">No conversation yet...</p>
                ) : (
                  history.map((h, i) => (
                    <div
                      key={h.id || i}
                      className={`p-4 rounded-2xl ${
                        h.sender === 'user' ? 'bg-slate-100' : 'bg-amber-50 border border-amber-100'
                      }`}
                    >
                      <span
                        className={`text-[9px] font-bold uppercase tracking-wider ${
                          h.sender === 'user' ? 'text-slate-400' : 'text-amber-600'
                        }`}
                      >
                        {h.sender === 'user' ? 'You' : 'Beatrice'}
                      </span>
                      <p className={`mt-1 text-sm ${h.sender === 'user' ? 'text-slate-600' : 'text-slate-800'}`}>
                        {h.text}
                      </p>
                    </div>
                  ))
                )}
              </div>

              <button
                onClick={clearMemory}
                className="mt-4 py-3 bg-slate-100 text-slate-500 rounded-full text-xs font-bold uppercase tracking-wider hover:bg-red-100 hover:text-red-600 transition-all"
              >
                Clear history
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
