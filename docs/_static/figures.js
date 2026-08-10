/* Make every themed figure clickable: the page shows the figure at the content-column
 * width, and a click opens the underlying SVG full-size in a new tab, where the browser
 * zoom works on it. Done here rather than in the themed-figure directive because the
 * final image URL only exists after Sphinx has copied and rewritten the asset paths --
 * img.src is that resolved URL, wherever the page sits in the site tree. */
(function () {
  "use strict";

  function makeClickable() {
    var images = document.querySelectorAll(
      "figure.only-light img, figure.only-dark img"
    );
    images.forEach(function (img) {
      if (img.parentNode.tagName === "A") return;
      var link = document.createElement("a");
      link.href = img.getAttribute("src");
      link.target = "_blank";
      link.rel = "noopener";
      link.title = "Open full size in a new tab";
      img.parentNode.insertBefore(link, img);
      link.appendChild(img);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", makeClickable);
  } else {
    makeClickable();
  }
})();
