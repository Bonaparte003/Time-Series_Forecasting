(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* Image lightbox */
  const lb = $("#lightbox");
  const lbImg = $("#lightbox-img");
  $$(".deck-fig img").forEach((img) => {
    img.addEventListener("click", (e) => {
      e.preventDefault();
      lbImg.src = img.src;
      lb.classList.add("is-open");
      lb.setAttribute("aria-hidden", "false");
    });
  });
  lb?.addEventListener("click", () => {
    lb.classList.remove("is-open");
    lb.setAttribute("aria-hidden", "true");
  });

  /* CSV modal — full untruncated table */
  const csvModal = $("#csv-modal");
  const csvTitle = $("#csv-modal-title");
  const csvMeta = $("#csv-modal-meta");
  const csvTable = $("#csv-modal-table");
  const csvClose = $("#csv-modal-close");
  let csvLoading = false;

  function closeCsvModal() {
    csvModal?.classList.remove("is-open");
    csvModal?.setAttribute("aria-hidden", "true");
    document.body.classList.remove("modal-open");
  }

  function renderCsvTable(headers, rows) {
    const headHtml = `<thead><tr>${headers
      .map((h) => `<th>${escapeHtml(h)}</th>`)
      .join("")}</tr></thead>`;
    const bodyHtml = `<tbody>${rows
      .map(
        (row) =>
          `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`
      )
      .join("")}</tbody>`;
    csvTable.innerHTML = headHtml + bodyHtml;
  }

  async function openCsvModal(path) {
    if (!csvModal || csvLoading) return;
    csvLoading = true;
    csvModal.classList.add("is-open", "is-loading");
    csvModal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
    csvTitle.textContent = path;
    csvMeta.textContent = "Loading…";
    csvTable.innerHTML = "";

    try {
      const res = await fetch(`/api/csv/?path=${encodeURIComponent(path)}`);
      if (!res.ok) throw new Error("not found");
      const data = await res.json();
      renderCsvTable(data.headers, data.rows);
      csvMeta.textContent = `${data.total_rows.toLocaleString()} rows · ${data.headers.length} columns`;
    } catch {
      csvMeta.textContent = "Could not load CSV.";
      csvTable.innerHTML = "";
    } finally {
      csvModal.classList.remove("is-loading");
      csvLoading = false;
    }
  }

  $$(".csv-panel.is-clickable").forEach((panel) => {
    const path = panel.dataset.csvPath;
    if (!path) return;

    panel.addEventListener("click", () => openCsvModal(path));
    panel.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openCsvModal(path);
      }
    });
  });

  csvClose?.addEventListener("click", (e) => {
    e.stopPropagation();
    closeCsvModal();
  });
  csvModal?.addEventListener("click", (e) => {
    if (e.target === csvModal) closeCsvModal();
  });
  csvModal?.querySelector(".csv-modal-dialog")?.addEventListener("click", (e) => {
    e.stopPropagation();
  });

  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    lb?.classList.remove("is-open");
    lb?.setAttribute("aria-hidden", "true");
    closeCsvModal();
  });
})();
