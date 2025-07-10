const overlap = document.querySelector('.overlap');
const len = overlap.textContent.length;

overlap.innerHTML = overlap.textContent.split('')
    .map((char, i) => `<span style="z-index: ${len - i}">${char}</span>`)
    .join('');