(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  /* Sticky nav highlight */
  const sections = $$("section[id]");
  const navLinks = $$('.side nav a[href^="#"]');

  function onScroll() {
    let current = sections[0]?.id;
    const y = window.scrollY + 80;
    for (const sec of sections) {
      if (sec.offsetTop <= y) current = sec.id;
    }
    navLinks.forEach((a) => {
      a.classList.toggle("is-active", a.getAttribute("href") === `#${current}`);
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* Square tabs — metrics */
  function wireTabs(groupName) {
    const tabs = $$(`[data-tab-group="${groupName}"] .tab`);
    const panels = $$(`[data-tab-panel="${groupName}"]`);
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const id = tab.dataset.tab;
        tabs.forEach((t) => t.classList.toggle("is-active", t === tab));
        panels.forEach((p) =>
          p.classList.toggle("is-active", p.dataset.tabId === id)
        );
        if (groupName === "forecasts") filterForecastPlots(id);
      });
    });
  }

  function filterForecastPlots(squareId) {
    $$(".plot-card").forEach((card) => {
      card.classList.toggle(
        "is-visible",
        card.dataset.square === String(squareId)
      );
    });
  }

  wireTabs("metrics");
  wireTabs("forecasts");
  const activeForecast = $('.tab[data-tab-group="forecasts"].is-active');
  if (activeForecast) filterForecastPlots(activeForecast.dataset.tab);

  /* Experiment table — fetch from SQLite API */
  const modelSel = $("#filter-model");
  const phaseSel = $("#filter-phase");
  const tbody = $("#exp-tbody");
  const status = $("#exp-status");

  async function loadExperiments() {
    if (!tbody) return;
    const params = new URLSearchParams();
    if (modelSel?.value) params.set("model", modelSel.value);
    if (phaseSel?.value) params.set("phase", phaseSel.value);
    params.set("limit", "40");
    status.textContent = "Loading…";
    try {
      const res = await fetch(`/api/experiments/?${params}`);
      const data = await res.json();
      tbody.innerHTML = data.rows
        .map(
          (r) => `
        <tr>
          <td>${r.id}</td>
          <td>${r.phase}</td>
          <td>${r.model}</td>
          <td>${r.val_mae}</td>
          <td>${r.val_mape}%</td>
          <td>${r.train_s}s</td>
        </tr>`
        )
        .join("");
      status.textContent = `${data.rows.length} runs · filtered via Django ORM → SQLite`;
    } catch (e) {
      status.textContent = "Could not load experiments. Run run_experiments first.";
      tbody.innerHTML = "";
    }
  }

  modelSel?.addEventListener("change", loadExperiments);
  phaseSel?.addEventListener("change", loadExperiments);
  loadExperiments();

  /* Lightbox for figures */
  const lb = $("#lightbox");
  const lbImg = $("#lightbox-img");
  $$("figure img").forEach((img) => {
    img.style.cursor = "zoom-in";
    img.addEventListener("click", (e) => {
      e.preventDefault();
      lbImg.src = img.src;
      lb.classList.add("is-open");
    });
  });
  lb?.addEventListener("click", () => lb.classList.remove("is-open"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") lb?.classList.remove("is-open");
  });
})();
