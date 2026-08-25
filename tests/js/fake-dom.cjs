class FakeClassList {
  constructor(node) { this.node = node; }
  contains(name) { return this.node.className.split(/\s+/).includes(name); }
  add(name) {
    if (!this.contains(name)) this.node.className = `${this.node.className} ${name}`.trim();
  }
  remove(name) {
    this.node.className = this.node.className.split(/\s+/)
      .filter((candidate) => candidate && candidate !== name).join(" ");
  }
}

class FakeElement {
  constructor(tagName, document) {
    this.tagName = tagName;
    this.ownerDocument = document;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.listeners = {};
    this.className = "";
    this._text = "";
    this.classList = new FakeClassList(this);
  }
  appendChild(child) {
    this.children.push(child);
    child.parentNode = this;
    if (this.tagName === "select" && (child.selected || this.value === undefined)) {
      this.value = child.value;
    }
    return child;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "class") this.className = String(value);
    if (name === "id") this.id = String(value);
  }
  getAttribute(name) { return name === "class" ? this.className : this.attributes[name] ?? null; }
  addEventListener(type, callback) { (this.listeners[type] ||= []).push(callback); }
  dispatch(type, event = {}) {
    event.key ||= "";
    event.preventDefault ||= () => {};
    for (const callback of this.listeners[type] || []) callback(event);
  }
  focus() { this.ownerDocument.activeElement = this; }
  querySelectorAll(selector) {
    const matches = [];
    const visit = (node) => {
      if (matchesSelector(node, selector)) matches.push(node);
      for (const child of node.children) visit(child);
    };
    for (const child of this.children) visit(child);
    return matches;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set innerHTML(value) { this._text = String(value); }
}

function matchesSelector(node, selector) {
  if (selector.startsWith(".")) {
    return selector.slice(1).split(".").every((name) => node.classList.contains(name));
  }
  const attribute = selector.match(/^\[([^=]+)="([^"]+)"\]$/);
  if (attribute) return node.getAttribute(attribute[1]) === attribute[2];
  return node.tagName === selector;
}

class FakeDocument {
  constructor() {
    this.baseURI = "https://example.test/downstream-susceptibility.html";
    this.readyState = "complete";
    this.body = new FakeElement("body", this);
    this.activeElement = this.body;
  }
  createElement(tagName) { return new FakeElement(tagName, this); }
  createElementNS(_namespace, tagName) { return new FakeElement(tagName, this); }
  querySelectorAll(selector) {
    const own = matchesSelector(this.body, selector) ? [this.body] : [];
    return own.concat(this.body.querySelectorAll(selector));
  }
}

module.exports = { FakeDocument };
