// ── State ────────────────────────────────────────────────────────────────────
const state = {
  selectedTopic: null,
  customTopic: '',
  questionCount: 5,
  sessionId: null,
  currentIndex: 0,
  totalQuestions: 0,
  lastEvaluation: null,
  eventSource: null,
};

// ── Screen management ────────────────────────────────────────────────────────
function showScreen(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}

// ── Screen 1: Topic Selection ────────────────────────────────────────────────
async function initTopicSelection() {
  const res = await fetch('/api/topics');
  const { topics } = await res.json();

  const container = document.getElementById('topic-chips');
  topics.forEach(topic => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = topic;
    chip.addEventListener('click', () => {
      container.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
      chip.classList.add('selected');
      state.selectedTopic = topic;
      state.customTopic = '';
      document.getElementById('custom-topic').value = '';
      updateStartButton();
    });
    container.appendChild(chip);
  });

  document.getElementById('custom-topic').addEventListener('input', e => {
    state.customTopic = e.target.value.trim();
    if (state.customTopic) {
      container.querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
      state.selectedTopic = null;
    }
    updateStartButton();
  });

  document.querySelectorAll('.count-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.count-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.questionCount = parseInt(btn.dataset.count, 10);
    });
  });

  document.getElementById('start-btn').addEventListener('click', startSession);
}

function updateStartButton() {
  const topic = state.selectedTopic || state.customTopic;
  document.getElementById('start-btn').disabled = !topic;
}

async function startSession() {
  const topic = state.selectedTopic || state.customTopic;
  document.getElementById('start-btn').disabled = true;
  document.getElementById('start-btn').textContent = 'Starting…';

  const res = await fetch('/api/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topic, question_count: state.questionCount }),
  });
  const data = await res.json();
  state.sessionId = data.session_id;
  state.totalQuestions = state.questionCount;
  state.currentIndex = 0;

  openSSEStream();

  document.getElementById('start-btn').textContent = 'Start Session →';
  document.getElementById('start-btn').disabled = false;
}

// ── SSE Stream ───────────────────────────────────────────────────────────────
function openSSEStream() {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = new EventSource(`/api/sessions/${state.sessionId}/stream`);

  state.eventSource.addEventListener('agent_status', e => {
    const { message } = JSON.parse(e.data);
    showAgentStatus(message);
  });

  state.eventSource.addEventListener('question_ready', e => {
    const { question, index } = JSON.parse(e.data);
    state.currentIndex = index;
    showQAScreen(question, index);
  });

  state.eventSource.addEventListener('evaluation_ready', e => {
    const { evaluation } = JSON.parse(e.data);
    state.lastEvaluation = evaluation;
    hideAgentStatus();
    document.getElementById('submit-btn').disabled = false;
    document.getElementById('answer-input').disabled = false;
    document.getElementById('answer-input').value = '';
  });

  state.eventSource.addEventListener('analysis_ready', e => {
    showAgentStatus('Building recommendations…');
  });

  state.eventSource.addEventListener('complete', e => {
    const { recommendation } = JSON.parse(e.data);
    state.eventSource.close();
    showResultsScreen(recommendation);
  });

  state.eventSource.addEventListener('error', e => {
    hideAgentStatus();
    alert('An error occurred. Please refresh and try again.');
  });
}

// ── Screen 2: Q&A Session ────────────────────────────────────────────────────
function showQAScreen(question, index) {
  showScreen('screen-qa');
  document.getElementById('qa-topic-label').textContent =
    `Topic: ${state.selectedTopic || state.customTopic}`;
  document.getElementById('qa-progress').textContent =
    `Q ${index + 1} of ${state.totalQuestions}`;
  document.getElementById('question-text').textContent = question.text;
  document.getElementById('difficulty-badge').textContent =
    `${difficultyIcon(question.difficulty)} ${capitalize(question.difficulty)} difficulty`;

  // Show previous evaluation if exists
  const prevCard = document.getElementById('prev-result');
  if (state.lastEvaluation && index > 0) {
    const ev = state.lastEvaluation;
    prevCard.innerHTML =
      `<span class="score">✓ Previous: ${ev.score}/100</span> — ${ev.feedback}`;
    prevCard.classList.remove('hidden');
  } else {
    prevCard.classList.add('hidden');
  }

  document.getElementById('answer-input').disabled = false;
  document.getElementById('answer-input').value = '';
  document.getElementById('submit-btn').disabled = false;
  hideAgentStatus();
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('submit-btn').addEventListener('click', submitAnswer);
  document.getElementById('new-session-btn').addEventListener('click', () => {
    state.sessionId = null;
    state.selectedTopic = null;
    state.customTopic = '';
    state.lastEvaluation = null;
    state.currentIndex = 0;
    document.getElementById('custom-topic').value = '';
    document.getElementById('topic-chips').querySelectorAll('.chip').forEach(c => c.classList.remove('selected'));
    updateStartButton();
    showScreen('screen-select');
  });
});

async function submitAnswer() {
  const text = document.getElementById('answer-input').value.trim();
  if (!text) return;

  document.getElementById('submit-btn').disabled = true;
  document.getElementById('answer-input').disabled = true;
  showAgentStatus('🤖 Submitting…');

  await fetch(`/api/sessions/${state.sessionId}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answer: text }),
  });
}

function showAgentStatus(msg) {
  const el = document.getElementById('agent-status');
  el.textContent = `🤖 ${msg}`;
  el.classList.remove('hidden');
}

function hideAgentStatus() {
  document.getElementById('agent-status').classList.add('hidden');
}

// ── Screen 3: Results ────────────────────────────────────────────────────────
async function showResultsScreen(recommendation) {
  // Load final session state for overall score
  const res = await fetch(`/api/sessions/${state.sessionId}`);
  const session = await res.json();
  const overallScore = session.analysis ? Math.round(session.analysis.overall_score) : '—';

  document.getElementById('overall-score').textContent = overallScore;

  const strengths = session.analysis?.strengths ?? [];
  const weaknesses = session.analysis?.weaknesses ?? [];

  renderList('strengths-list', strengths);
  renderList('weaknesses-list', weaknesses);

  document.getElementById('advisor-message').textContent = recommendation.message;

  const nextTopicsEl = document.getElementById('next-topics');
  nextTopicsEl.innerHTML = '';
  recommendation.next_topics.forEach(topic => {
    const chip = document.createElement('button');
    chip.className = 'chip';
    chip.textContent = topic;
    chip.addEventListener('click', () => {
      state.selectedTopic = topic;
      state.customTopic = '';
      showScreen('screen-select');
      updateStartButton();
    });
    nextTopicsEl.appendChild(chip);
  });

  renderList('focus-areas', recommendation.focus_areas);
  showScreen('screen-results');
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function renderList(id, items) {
  const el = document.getElementById(id);
  el.innerHTML = items.map(item => `<li>${item}</li>`).join('');
}

function difficultyIcon(d) {
  return d === 'easy' ? '🟢' : d === 'medium' ? '⚡' : '🔴';
}

function capitalize(s) {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ── Init ──────────────────────────────────────────────────────────────────────
initTopicSelection();
