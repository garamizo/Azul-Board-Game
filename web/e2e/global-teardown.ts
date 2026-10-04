import { execSync } from 'node:child_process';

export default function teardown() {
  execSync('make -C .. e2e-server-stop', { stdio: 'inherit' });
}
