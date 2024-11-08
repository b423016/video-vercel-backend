import dotenv from 'dotenv';
dotenv.config();

export const SERVER_PORT = process.env.PORT || 5000;
export const UPLOAD_DIR = 'uploads';
