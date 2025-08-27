// server.js
import express from 'express';
import multer from 'multer';
import { Anthropic } from '@anthropic-ai/sdk';
import AWS from 'aws-sdk';
import { SpeechClient } from '@google-cloud/speech';

const app = express();
const upload = multer();

// Initialize APIs
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const polly = new AWS.Polly({
  region: 'us-west-2',
  accessKeyId: process.env.AWS_ACCESS_KEY,
  secretAccessKey: process.env.AWS_SECRET_KEY
});
const speechClient = new SpeechClient({
  keyFilename: 'path/to/google-credentials.json'
});

// Speech to Text
async function convertSpeechToText(audioBuffer, language) {
  const audio = {
    content: audioBuffer.toString('base64')
  };
  const config = {
    encoding: 'LINEAR16',
    sampleRateHertz: 16000,
    languageCode: language,
    enableAutomaticPunctuation: true,
    model: 'latest_long'
  };

  const [response] = await speechClient.recognize({
    config,
    audio
  });

  return response.results
    .map(result => result.alternatives[0].transcript)
    .join('\n');
}

// Text Translation with Claude
async function translateText(text, sourceLang, targetLang) {
  const completion = await anthropic.messages.create({
    model: "claude-3-opus-20240229",
    max_tokens: 1024,
    messages: [{
      role: "user", 
      content: `Translate this text from ${sourceLang} to ${targetLang}, preserving tone and context: ${text}`
    }]
  });
  return completion.content;
}

// Text to Speech with Amazon Polly
async function convertTextToSpeech(text, language) {
  const params = {
    Engine: 'neural',
    LanguageCode: language,
    Text: text,
    OutputFormat: 'mp3',
    VoiceId: getVoiceForLanguage(language)
  };

  const response = await polly.synthesizeSpeech(params).promise();
  return response.AudioStream;
}

// Helper function to get appropriate voice for each language
function getVoiceForLanguage(language) {
  const voices = {
    'en-US': 'Matthew',
    'es-ES': 'Lucia',
    'fr-FR': 'Lea',
    'de-DE': 'Daniel',
    'ja-JP': 'Kazuha',
    'zh-CN': 'Zhiyu'
    // Add more language-voice mappings
  };
  return voices[language] || 'Joanna';
}

// Main translation endpoint
app.post('/translate', upload.single('audio'), async (req, res) => {
  try {
    const { sourceLang, targetLang } = req.body;

    // 1. Speech to Text
    const transcribedText = await convertSpeechToText(
      req.file.buffer,
      sourceLang
    );

    // 2. Translate Text
    const translatedText = await translateText(
      transcribedText,
      sourceLang,
      targetLang
    );

    // 3. Text to Speech
    const audioContent = await convertTextToSpeech(
      translatedText,
      targetLang
    );

    res.json({
      success: true,
      originalText: transcribedText,
      translatedText: translatedText,
      audioContent: audioContent.toString('base64')
    });

  } catch (error) {
    console.error('Translation error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'healthy' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  const errorMessage = errorHandler[err.type] || 'An unexpected error occurred';
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    error: errorMessage, code: err.type
  });
});

// Add to server.js
const errorHandler = {
    AudioError: 'Check microphone permissions and connection',
    NetworkError: 'Please check your internet connection',
    TranslationError: 'Translation service temporarily unavailable',
    QuotaExceeded: 'Usage limit reached'
};

app.use((err, req, res, next) => {
    const errorMessage = errorHandler[err.type] || 'An unexpected error occurred';
    res.status(500).json({ error: errorMessage, code: err.type });
 });