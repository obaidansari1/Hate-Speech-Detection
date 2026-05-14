async function startAnalysis() {
    const textInput = document.getElementById('input').value;
    if (!textInput) return;

    const btn = document.getElementById('analyze-btn');
    const loader = document.getElementById('loading');
    const resArea = document.getElementById('results');

    btn.disabled = true;
    loader.style.display = 'block';
    resArea.style.display = 'none';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: textInput })
        });

        const data = await response.json();

        resArea.innerHTML = `
<div class="result-card reveal in">

    <div class="rc-head">
        <span class="rc-label">Verdict</span>
    </div>

    <div class="rc-body">
        <div class="verdict-big ${data.verdict.toLowerCase().split(' ')[0]}">
            <div class="v-label">${data.verdict}</div>
        </div>
    </div>

    <div class="bars">

        <div class="bar">
            <span>Hate</span>
            <div class="progress">
                <div class="fill red" style="width:${data.scores.hate}%"></div>
            </div>
            <span>${data.scores.hate}%</span>
        </div>

        <div class="bar">
            <span>Offensive</span>
            <div class="progress">
                <div class="fill orange" style="width:${data.scores.off}%"></div>
            </div>
            <span>${data.scores.off}%</span>
        </div>

        <div class="bar">
            <span>Neutral</span>
            <div class="progress">
                <div class="fill green" style="width:${data.scores.neu}%"></div>
            </div>
            <span>${data.scores.neu}%</span>
        </div>

    </div>

</div>
`;
        resArea.style.display = 'flex';
    } catch (e) {
        console.error("Error:", e);
    } finally {
        loader.style.display = 'none';
        btn.disabled = false;
    }
}

// Animation Observer
document.addEventListener("DOMContentLoaded", () => {
    const obs = new IntersectionObserver((es) => {
        es.forEach(e => { if(e.isIntersecting) e.target.classList.add('in'); });
    }, {threshold: 0.1});
    document.querySelectorAll('.reveal, .reveal-left').forEach(el => obs.observe(el));
});