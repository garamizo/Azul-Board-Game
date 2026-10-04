import { mount } from 'svelte';
import App from './App.svelte';
import './app.css';

// Dev mode only: /?as=bob@example.com picks who you are (the server ignores
// the cookie unless it runs without Access).
const as = new URLSearchParams(location.search).get('as');
if (as) document.cookie = `azul_dev_user=${encodeURIComponent(as)}; path=/; SameSite=Strict`;

mount(App, { target: document.getElementById('app')! });
