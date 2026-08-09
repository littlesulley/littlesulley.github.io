/* exported searchFunc */
(function() {
  "use strict";

  function collapseWhitespace(value) {
    return String(value || "").replace(/\s+/g, " ").trim();
  }

  function normalize(value) {
    return collapseWhitespace(value).toLocaleLowerCase();
  }

  function formatCount(template, count) {
    return count + " " + String(template || "results");
  }

  function createResult(item, keywords) {
    var li = document.createElement("li");
    var link = document.createElement("a");
    var title = document.createElement("span");
    var excerpt = document.createElement("p");
    var content = collapseWhitespace(item.content);
    var normalizedContent = normalize(content);
    var firstOccurrence = -1;

    keywords.some(function(keyword) {
      firstOccurrence = normalizedContent.indexOf(keyword);
      return firstOccurrence >= 0;
    });

    if (firstOccurrence < 0) {
      firstOccurrence = 0;
    }

    var start = Math.max(0, firstOccurrence - 45);
    var end = Math.min(content.length, firstOccurrence + 155);
    var snippet = content.slice(start, end);

    if (start > 0) {
      snippet = "…" + snippet;
    }
    if (end < content.length) {
      snippet += "…";
    }

    link.className = "search-result-title";
    link.href = item.url;
    title.textContent = item.title || "Untitled";
    link.appendChild(title);
    li.appendChild(link);

    if (snippet) {
      excerpt.className = "search-result";
      excerpt.textContent = snippet;
      li.appendChild(excerpt);
    }

    return li;
  }

  function searchFunc(path, searchId, contentId, statusId, labels) {
    var input = document.getElementById(searchId);
    var resultContent = document.getElementById(contentId);
    var status = document.getElementById(statusId);
    var data = [];
    var loaded = false;

    if (!input || !resultContent || !status) {
      return;
    }

    status.dataset.searchInitialized = "true";

    function setStatus(message, isError) {
      status.textContent = message;
      status.classList.toggle("search-error", Boolean(isError));
    }

    function runSearch() {
      var query = normalize(input.value);
      var keywords = query.split(/\s+/).filter(Boolean);

      resultContent.replaceChildren();

      if (!query) {
        setStatus(loaded ? labels.ready : labels.loading);
        return;
      }

      if (!loaded) {
        setStatus(labels.loading);
        return;
      }

      var results = data.map(function(item) {
        var title = normalize(item.title);
        var tags = normalize((item.tags || []).join(" "));
        var categories = normalize((item.categories || []).join(" "));
        var content = normalize(item.content);
        var searchable = [title, tags, categories, content].join(" ");

        if (!keywords.every(function(keyword) { return searchable.indexOf(keyword) >= 0; })) {
          return null;
        }

        var score = keywords.reduce(function(total, keyword) {
          return total
            + (title.indexOf(keyword) >= 0 ? 8 : 0)
            + (tags.indexOf(keyword) >= 0 ? 4 : 0)
            + (categories.indexOf(keyword) >= 0 ? 2 : 0)
            + (content.indexOf(keyword) >= 0 ? 1 : 0);
        }, 0);

        return { item: item, score: score };
      }).filter(Boolean).sort(function(a, b) {
        return b.score - a.score;
      }).slice(0, 30);

      if (!results.length) {
        setStatus(labels.noResults);
        return;
      }

      var list = document.createElement("ul");
      list.className = "search-result-list";

      results.forEach(function(result) {
        list.appendChild(createResult(result.item, keywords));
      });

      resultContent.appendChild(list);
      setStatus(formatCount(labels.resultCount, results.length));
    }

    input.addEventListener("input", runSearch);
    input.addEventListener("keydown", function(event) {
      if (event.key === "Enter") {
        event.preventDefault();
      }
    });

    setStatus(labels.loading);

    function indexLoaded(index) {
        data = Array.isArray(index) ? index : [];
        loaded = true;
        runSearch();
    }

    function indexFailed() {
      setStatus(labels.loadFailed, true);
    }

    if (typeof window.fetch === "function") {
      window.fetch(path, { credentials: "same-origin" })
        .then(function(response) {
          if (!response.ok) {
            throw new Error("Search index request failed");
          }
          return response.json();
        })
        .then(indexLoaded)
        .catch(indexFailed);
    } else {
      var request = new XMLHttpRequest();
      request.open("GET", path, true);
      request.onreadystatechange = function() {
        if (request.readyState !== 4) {
          return;
        }
        if (request.status >= 200 && request.status < 300) {
          try {
            indexLoaded(JSON.parse(request.responseText));
          } catch (error) {
            indexFailed();
          }
        } else {
          indexFailed();
        }
      };
      request.onerror = indexFailed;
      request.send();
    }
  }

  window.searchFunc = searchFunc;
}());
