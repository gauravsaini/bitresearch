/// <reference types="@webgpu/types" />

// Vite raw import for .wgsl files
declare module '*.wgsl?raw' {
  const content: string;
  export default content;
}
