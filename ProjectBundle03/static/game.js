


let cards = [];
let revealed = [];
let lockBoard = false;

let playerTurn = 1;
let scores = {1: 0, 2: 0};

function updateScoreboard() {
    document.getElementById("p1").innerText = scores[1];
    document.getElementById("p2").innerText = scores[2];
    document.getElementById("turn").innerText = "Turn: Player " + playerTurn;
}

function switchPlayer() {
    playerTurn = playerTurn === 1 ? 2 : 1;
    updateScoreboard();
}

function revealCard(cardElem, cardData) {
    if (lockBoard) return;
    if (cardElem.classList.contains("revealed")) return;

    cardElem.classList.add("revealed");
    revealed.push({elem: cardElem, data: cardData});

    if (revealed.length === 2) {
        lockBoard = true;
        checkMatch();
    }
}

function checkMatch() {
    const [c1, c2] = revealed;

    if (c1.data.pair_key === c2.data.pair_key && c1.data.type !== c2.data.type) {
        // MATCH
        scores[playerTurn]++;
        updateScoreboard();

        c1.elem.classList.add("matched");
        c2.elem.classList.add("matched");

        setTimeout(() => {
            c1.elem.classList.remove("matched");
            c2.elem.classList.remove("matched");
        }, 2500);

        revealed = [];
        lockBoard = false;
    } else {
        setTimeout(() => {
            c1.elem.classList.remove("revealed");
            c2.elem.classList.remove("revealed");

            revealed = [];
            lockBoard = false;
            switchPlayer();
        }, 1000);
    }
}

function createBoard(cards) {
    const board = document.getElementById("gameboard");
    board.innerHTML = "";

    cards.forEach(card => {
        const div = document.createElement("div");
        div.className = "card";

        const inner = document.createElement("div");
        inner.className = "card-inner";

        const front = document.createElement("div");
        front.className = "card-front";

        const back = document.createElement("div");
        back.className = "card-back";

        const img = document.createElement("img");
        img.src = card.file;

        back.appendChild(img);
        inner.appendChild(front);
        inner.appendChild(back);
        div.appendChild(inner);

        div.addEventListener("click", () => revealCard(div, card));

        board.appendChild(div);
    });
}

function resetGame() {
    scores = {1: 0, 2: 0};
    playerTurn = 1;

    fetch("/cards")
        .then(res => res.json())
        .then(data => {
            cards = data;
            createBoard(cards);
            updateScoreboard();
            revealed = [];
            lockBoard = false;
        });
}

document.getElementById("resetBtn").addEventListener("click", resetGame);

fetch("/cards")
    .then(res => res.json())
    .then(data => {
        cards = data;
        createBoard(cards);
        updateScoreboard();
    });
