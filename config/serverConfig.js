import { config } from 'dotenv';
config();

export const PORT = process.env.PORT || 3000;
export const CORS_ORIGIN = process.env.CORS_ORIGIN || 'https://video-vercel.vercel.app';