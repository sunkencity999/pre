// Tests for eGPU (TinyGPU) detection and support across macOS scripts
// Validates that install.sh, pre-server.sh, and pre-launch handle eGPU correctly.

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..', '..');

describe('eGPU support in install.sh', () => {
  const installPath = path.join(REPO_ROOT, 'install.sh');
  let content;

  beforeAll(() => {
    content = fs.readFileSync(installPath, 'utf-8');
  });

  test('detects TinyGPU driver extension', () => {
    expect(content).toContain('systemextensionsctl');
    expect(content).toContain('tinygpu');
  });

  test('checks for TinyGPU.app', () => {
    expect(content).toContain('TinyGPU.app');
  });

  test('detects Thunderbolt GPU devices', () => {
    expect(content).toContain('SPThunderboltDataType');
  });

  test('detects NVIDIA eGPU VRAM via Docker nvidia-smi', () => {
    expect(content).toContain('docker');
    expect(content).toContain('nvidia-smi');
    expect(content).toContain('memory.total');
    expect(content).toContain('EGPU_VRAM_GB');
  });

  test('supports both Apple Silicon and Intel architectures', () => {
    expect(content).toContain('IS_APPLE_SILICON');
    expect(content).toContain('IS_INTEL');
    expect(content).toContain('x86_64');
    expect(content).toContain('arm64');
  });

  test('requires eGPU for Intel Macs', () => {
    expect(content).toContain('Intel Macs require an eGPU');
    expect(content).toContain('docs.tinygrad.org/tinygpu');
  });

  test('uses VRAM-based quant selection for eGPU systems', () => {
    expect(content).toContain('EGPU_VRAM_GB');
    expect(content).toContain('q4_K_M');
    expect(content).toContain('q8_0');
    // Checks VRAM >= 28 for q8_0
    expect(content).toMatch(/EGPU_VRAM_GB.*-ge 28/);
  });

  test('supports headroom-based context window sizing', () => {
    expect(content).toContain('HEADROOM');
    expect(content).toContain('MODEL_SIZE_GB');
    // Same thresholds as Windows installer
    expect(content).toContain('131072');
    expect(content).toContain('65536');
    expect(content).toContain('32768');
    expect(content).toContain('16384');
    expect(content).toContain('8192');
  });

  test('sets GPU_BACKEND to cuda for NVIDIA eGPU', () => {
    expect(content).toContain('GPU_BACKEND="cuda"');
    expect(content).toContain('GPU_BACKEND="metal"');
  });

  test('sets Flash Attention based on GPU backend', () => {
    // FA=0 for Metal, FA=1 for CUDA
    expect(content).toContain('FA_VAL=0');
    expect(content).toContain('FA_VAL=1');
  });

  test('sets KV cache type based on quant', () => {
    expect(content).toContain('KV_CACHE_TYPE');
    expect(content).toContain('OLLAMA_KV_CACHE_TYPE');
  });

  test('checks Docker availability for NVIDIA eGPU', () => {
    expect(content).toContain('Docker Desktop not found');
    expect(content).toContain('docker.com');
  });

  test('relaxes RAM minimum for Intel eGPU systems', () => {
    // Intel eGPU: 16GB minimum (model lives on VRAM)
    // Apple Silicon: 32GB minimum (unified memory)
    expect(content).toMatch(/IS_INTEL.*16/s);
    expect(content).toMatch(/IS_APPLE_SILICON.*32/s);
  });

  test('skips CLI build on Intel', () => {
    expect(content).toContain('Skipping CLI build');
    expect(content).toContain('Web GUI only');
  });

  test('patches Modelfile FROM line for q4_K_M', () => {
    expect(content).toContain('EFFECTIVE_MODELFILE');
    expect(content).toContain('FROM gemma4:26b-a4b-it-q8_0');
    expect(content).toContain('TEMP_MODELFILE');
  });

  test('lowers macOS version requirement for Intel eGPU', () => {
    // Apple Silicon: macOS 14+, Intel eGPU: macOS 12.1+
    expect(content).toMatch(/IS_APPLE_SILICON.*14/s);
    expect(content).toMatch(/IS_INTEL.*12/s);
  });
});

describe('eGPU support in pre-server.sh', () => {
  const serverPath = path.join(REPO_ROOT, 'web', 'pre-server.sh');
  let content;

  beforeAll(() => {
    content = fs.readFileSync(serverPath, 'utf-8');
  });

  test('detects TinyGPU at startup', () => {
    expect(content).toContain('systemextensionsctl');
    expect(content).toContain('tinygpu');
  });

  test('sets Flash Attention based on GPU backend', () => {
    expect(content).toContain('OLLAMA_FLASH_ATTENTION=1');
    expect(content).toContain('OLLAMA_FLASH_ATTENTION=0');
  });

  test('detects VRAM via Docker for KV cache type', () => {
    expect(content).toContain('nvidia-smi');
    expect(content).toContain('EGPU_VRAM');
    expect(content).toContain('OLLAMA_KV_CACHE_TYPE');
  });

  test('sets GPU overhead for CUDA backend', () => {
    expect(content).toContain('OLLAMA_GPU_OVERHEAD');
  });
});

describe('eGPU detection in pre-launch', () => {
  const launchPath = path.join(REPO_ROOT, 'engine', 'pre-launch');
  let content;

  beforeAll(() => {
    content = fs.readFileSync(launchPath, 'utf-8');
  });

  test('detects TinyGPU for status display', () => {
    expect(content).toContain('tinygpu');
    expect(content).toContain('EGPU_STATUS');
  });
});

describe('cross-script consistency', () => {
  test('install.sh and install.ps1 use same headroom thresholds', () => {
    const installSh = fs.readFileSync(path.join(REPO_ROOT, 'install.sh'), 'utf-8');
    const installPs1 = fs.readFileSync(path.join(REPO_ROOT, 'install.ps1'), 'utf-8');

    // Both should use headroom-based context sizing with same breakpoints
    for (const threshold of ['131072', '65536', '32768', '16384', '8192']) {
      expect(installSh).toContain(threshold);
      expect(installPs1).toContain(threshold);
    }
  });

  test('install.sh and install.ps1 both detect VRAM for quant selection', () => {
    const installSh = fs.readFileSync(path.join(REPO_ROOT, 'install.sh'), 'utf-8');
    const installPs1 = fs.readFileSync(path.join(REPO_ROOT, 'install.ps1'), 'utf-8');

    // Both should reference VRAM in quant selection
    expect(installSh).toContain('VRAM');
    expect(installPs1).toContain('vramGB');
    // Both should have q4_K_M and q8_0 paths
    expect(installSh).toContain('q4_K_M');
    expect(installPs1).toContain('q4_K_M');
  });

  test('pre-server.sh and pre-server.ps1 both detect VRAM for KV cache', () => {
    const serverSh = fs.readFileSync(path.join(REPO_ROOT, 'web', 'pre-server.sh'), 'utf-8');
    const serverPs1 = fs.readFileSync(path.join(REPO_ROOT, 'web', 'pre-server.ps1'), 'utf-8');

    expect(serverSh).toContain('OLLAMA_KV_CACHE_TYPE');
    expect(serverPs1).toContain('OLLAMA_KV_CACHE_TYPE');
    expect(serverSh).toContain('OLLAMA_FLASH_ATTENTION');
    expect(serverPs1).toContain('OLLAMA_FLASH_ATTENTION');
  });
});
