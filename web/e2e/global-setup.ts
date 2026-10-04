import { execSync } from 'node:child_process';

export default function setup() {
  execSync('make -C .. e2e-server-start', { stdio: 'inherit' });
}
