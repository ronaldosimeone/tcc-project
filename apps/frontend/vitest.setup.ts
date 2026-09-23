import "@testing-library/jest-dom";
import "jest-axe/extend-expect";

// jsdom não implementa ResizeObserver — necessário pelo Radix Slider
// (RF-25, `@radix-ui/react-use-size`) e por qualquer outro primitivo Radix
// que meça o próprio elemento. Stub mínimo, sem comportamento real (nenhum
// teste depende de callbacks de resize disparando de verdade).
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver =
    ResizeObserverStub as unknown as typeof ResizeObserver;
}
