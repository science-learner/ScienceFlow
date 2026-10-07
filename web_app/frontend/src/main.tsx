import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConfigProvider, App as AntApp, theme } from 'antd';
import App from './App';
import './styles/variables.css';
import './styles/base.css';
import './styles/panels.css';
import './styles/components.css';
import './styles/chat.css';
import './styles/workspace.css';
import './styles/agent-map.css';
import './styles/doc.css';
import './styles/overrides.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#1677ff',
          borderRadius: 8,
          colorBgContainer: 'rgba(15,20,28,0.88)',
          colorBgElevated: '#151d28',
          colorBorder: 'rgba(255,255,255,0.08)',
          colorText: '#d9e2ec',
          colorTextSecondary: '#aab7c6',
          fontFamily: "var(--sans)",
        },
      }}
    >
      <AntApp>
        <App />
      </AntApp>
    </ConfigProvider>
  </React.StrictMode>
);
