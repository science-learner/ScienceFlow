import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, Form, Input, Button, Typography, App } from 'antd';
import { UserOutlined, LockOutlined, LoginOutlined } from '@ant-design/icons';
import { Languages } from 'lucide-react';
import * as api from '../api/client';
import { useAppStore } from '../store/useAppStore';
import { useT } from '../i18n/useT';
import type { Lang } from '../i18n/translations';
import type { AuthUser } from '../types';

const { Title, Text } = Typography;

interface LoginPageProps {
  onSuccess: (token: string, user: AuthUser, userName: string, password: string) => void;
}

export function LoginPage({ onSuccess }: LoginPageProps) {
  const [loading, setLoading] = useState(false);
  const [form] = Form.useForm();
  const { message: msgApi } = App.useApp();
  const { language, setLanguage } = useAppStore();
  const t = useT();

  const toggleLang = useCallback(() => {
    const next: Lang = language === 'en-US' ? 'zh-CN' : 'en-US';
    setLanguage(next);
  }, [language, setLanguage]);

  const canvasRef = useRef<HTMLCanvasElement>(null);

  const drawParticles = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return () => {};
    const ctx = canvas.getContext('2d');
    if (!ctx) return () => {};

    let animId = 0;
    const particles: { x: number; y: number; vx: number; vy: number; r: number; a: number; da: number }[] = [];
    const count = 60;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.2 + 0.4,
        a: Math.random(),
        da: (Math.random() - 0.5) * 0.012,
      });
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = 'rgba(22,119,255,0.04)';
      ctx.lineWidth = 0.5;
      const grid = 56;
      for (let x = grid; x < canvas.width; x += grid) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = grid; y < canvas.height; y += grid) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }

      for (const p of particles) {
        p.x += p.vx; p.y += p.vy; p.a += p.da;
        if (p.x < 0) p.x = canvas.width; if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;
        const alpha = 0.25 + 0.6 * Math.abs(Math.sin(p.a));
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(22,119,255,${alpha})`; ctx.fill();
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(22,119,255,${alpha * 0.06})`; ctx.fill();
      }

      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 130) {
            ctx.beginPath(); ctx.moveTo(particles[i].x, particles[i].y); ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(22,119,255,${0.03 * (1 - dist / 130)})`;
            ctx.lineWidth = 0.3; ctx.stroke();
          }
        }
      }
      animId = requestAnimationFrame(draw);
    };
    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  useEffect(() => {
    const cleanup = drawParticles();
    return cleanup;
  }, [drawParticles]);

  const handleLogin = useCallback(async (values: { username: string; password: string }) => {
    setLoading(true);
    try {
      const result = await api.login(values.username, values.password);
      msgApi.success(t.login.welcome);
      setTimeout(() => onSuccess(result.token, result.user, values.username, values.password), 400);
    } catch (e: unknown) {
      msgApi.error((e as Error).message || t.login.loginFailed);
    } finally {
      setLoading(false);
    }
  }, [onSuccess, msgApi, t]);

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 10000,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(22,119,255,0.06), transparent 70%), radial-gradient(ellipse 60% 40% at 80% 80%, rgba(22,119,255,0.03), transparent 70%), #0a0d14',
      overflow: 'hidden',
    }}>
      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} />

      <button
        className="login-lang-toggle"
        type="button"
        title={language === 'en-US' ? t.topbar.switchToChinese : t.topbar.switchToEnglish}
        aria-label={language === 'en-US' ? t.topbar.switchToChinese : t.topbar.switchToEnglish}
        onClick={toggleLang}
      >
        <Languages size={14} />
        {language === 'en-US' ? '中文' : 'EN'}
      </button>

      <div style={{ position: 'relative', zIndex: 1, width: 380, maxWidth: '92vw' }}>
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <div style={{
            width: 64, height: 64, margin: '0 auto 16px',
            borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'linear-gradient(135deg, rgba(22,119,255,0.12), rgba(22,119,255,0.04))',
            border: '1px solid rgba(22,119,255,0.15)',
            overflow: 'hidden',
          }}>
            <img
              src="/favicon.ico"
              alt="ScienceFlow"
              style={{ width: 40, height: 40, objectFit: 'contain' }}
            />
          </div>
          <Title level={2} style={{ margin: 0, color: '#e8f0fe', letterSpacing: -0.5 }}>
            ScienceFlow
          </Title>
          <Text type="secondary" style={{ fontFamily: 'var(--mono)', fontSize: 12, letterSpacing: 2, textTransform: 'uppercase' }}>
            {t.login.tagline1}
          </Text>
          <br></br>
          <Text type="secondary" style={{ fontFamily: 'var(--mono)', fontSize: 12, letterSpacing: 2, textTransform: 'uppercase' }}>
            {t.login.tagline2}
          </Text>
        </div>

        <Card
          styles={{ body: { padding: '28px 28px 24px' } }}
          style={{
            border: '1px solid rgba(22,119,255,0.1)',
            borderRadius: 16,
            background: 'rgba(15,20,28,0.88)',
            backdropFilter: 'blur(24px)',
            boxShadow: '0 0 0 1px rgba(22,119,255,0.04), 0 16px 64px rgba(0,0,0,0.5), 0 0 120px rgba(22,119,255,0.03)',
          }}
        >
          <Title level={5} style={{ textAlign: 'center', margin: '0 0 20px', color: 'rgba(255,255,255,0.55)', fontWeight: 400, fontFamily: 'var(--mono)', fontSize: 13, letterSpacing: 1, textTransform: 'uppercase' }}>
            {t.login.authRequired}
          </Title>

          <Form form={form} onFinish={handleLogin} layout="vertical" size="large" autoComplete="off">
            <Form.Item name="username" rules={[{ required: true, message: t.login.usernameRequired }]}>
              <Input prefix={<UserOutlined style={{ color: 'rgba(255,255,255,0.25)' }} />} placeholder={t.login.usernamePlaceholder} />
            </Form.Item>
            <Form.Item name="password" rules={[{ required: true, message: t.login.passwordRequired }]}>
              <Input.Password prefix={<LockOutlined style={{ color: 'rgba(255,255,255,0.25)' }} />} placeholder={t.login.passwordPlaceholder} />
            </Form.Item>
            <Form.Item style={{ marginBottom: 0 }}>
              <Button type="primary" htmlType="submit" loading={loading} block size="large" icon={<LoginOutlined />}>
                {t.login.signIn}
              </Button>
            </Form.Item>
          </Form>

          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <Text type="secondary" style={{ fontSize: 11, fontFamily: 'var(--mono)' }}>
              {t.login.cachedHint}
            </Text>
          </div>
        </Card>

        <div style={{
          marginTop: 20, textAlign: 'center',
          fontFamily: 'var(--mono)', fontSize: 10, color: 'rgba(255,255,255,0.1)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
        }}>
          <span>ScienceFlow v0.1</span>
          <span style={{ width: 3, height: 3, borderRadius: '50%', background: 'rgba(255,255,255,0.1)' }} />
          <span>{t.login.engineName}</span>
        </div>
      </div>
    </div>
  );
}
