// service-worker.js - FinanceClick PWA CORRIGIDO
const CACHE_NAME = 'financeclick-v3.0.0';
const STATIC_CACHE = 'static-v3';
const DYNAMIC_CACHE = 'dynamic-v3';

// Arquivos para cache estático - CAMINHOS CORRETOS
const STATIC_FILES = [
  '/',
  '/index.html',
  '/dashboard.html',
  '/history.html', 
  '/guide.html',
  '/about.html',
  '/contact.html',
  '/style.css',
  '/script.js',
  '/manifest.json',
  '/offline.html',
  
  // Ícones com caminhos corrigidos
  '/frontend/icons/icon-72x72.png',
  '/frontend/icons/icon-96x96.png',
  '/frontend/icons/icon-128x128.png',
  '/frontend/icons/icon-144x144.png',
  '/frontend/icons/icon-152x152.png',
  '/frontend/icons/icon-192x192.png',
  '/frontend/icons/icon-384x384.png',
  '/frontend/icons/icon-512x512.png'
];

// Instalação - Cache dos arquivos estáticos
self.addEventListener('install', (event) => {
  console.log('🚀 Service Worker instalando...');
  
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => {
        console.log('📦 Cacheando arquivos estáticos');
        return cache.addAll(STATIC_FILES);
      })
      .then(() => {
        console.log('✅ Service Worker instalado');
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('❌ Erro na instalação do Service Worker:', error);
        return self.skipWaiting();
      })
  );
});

// Ativação - Limpar caches antigos
self.addEventListener('activate', (event) => {
  console.log('🔄 Service Worker ativando...');
  
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== STATIC_CACHE && cacheName !== DYNAMIC_CACHE) {
            console.log('🧹 Removendo cache antigo:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('✅ Service Worker ativado');
      return self.clients.claim();
    })
  );
});

// Estratégia: Cache First com fallback para network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Ignorar requisições para a API (sempre network)
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/auth/')) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Para arquivos estáticos: Cache First
  event.respondWith(cacheFirst(request));
});

// Estratégia: Cache First
async function cacheFirst(request) {
  try {
    const cachedResponse = await caches.match(request);
    
    if (cachedResponse) {
      return cachedResponse;
    }

    const networkResponse = await fetch(request);
    
    if (networkResponse.ok) {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('🌐 Offline - Servindo página offline');
    
    if (request.destination === 'document') {
      const offlinePage = await caches.match('/offline.html');
      if (offlinePage) {
        return offlinePage;
      }
    }
    
    return new Response('Offline', { 
      status: 503,
      statusText: 'Service Unavailable'
    });
  }
}

// Estratégia: Network First para APIs
async function networkFirst(request) {
  try {
    const networkResponse = await fetch(request);
    
    if (networkResponse.ok && request.method === 'GET') {
      const cache = await caches.open(DYNAMIC_CACHE);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.log('📡 API offline - Tentando cache');
    
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    return new Response(
      JSON.stringify({ 
        error: 'Connection lost',
        message: 'Please check your internet connection',
        code: 'NETWORK_ERROR'
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }
}

// Sincronização em background
self.addEventListener('sync', (event) => {
  if (event.tag === 'background-sync') {
    console.log('🔄 Sincronização em background');
    event.waitUntil(doBackgroundSync());
  }
});

async function doBackgroundSync() {
  try {
    console.log('📡 Sincronizando dados em background...');
    // Implementar lógica de sincronização aqui
  } catch (error) {
    console.error('❌ Erro na sincronização:', error);
  }
}

// Notificações push
self.addEventListener('push', (event) => {
  if (!event.data) return;

  try {
    const data = event.data.json();
    const options = {
      body: data.body || 'Nova notificação do FinanceClick',
      icon: '/frontend/icons/icon-192x192.png',
      badge: '/frontend/icons/icon-72x72.png',
      vibrate: [100, 50, 100],
      data: {
        url: data.url || '/'
      },
      actions: [
        {
          action: 'open',
          title: 'Abrir'
        },
        {
          action: 'close', 
          title: 'Fechar'
        }
      ]
    };

    event.waitUntil(
      self.registration.showNotification(data.title || 'FinanceClick', options)
    );
  } catch (error) {
    console.error('❌ Erro na notificação push:', error);
  }
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  if (event.action === 'open') {
    event.waitUntil(
      clients.openWindow(event.notification.data.url)
    );
  }
});

// Gerenciamento de mensagens
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});