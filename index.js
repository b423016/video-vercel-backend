import express from 'express';
import cors from 'cors';
import path from 'path';
import fs from 'fs';
import uploadRoutes from './routes/uploadRoutes.js';
import { PORT } from './config/serverConfig.js';

const app = express();

const allowedOrigins = ['http://localhost:5173'];
const corsOptions = {
    origin: (origin, callback) => {
        console.log("Request Origin:", origin); 
        if (!origin || allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
};

app.use(cors(corsOptions));

const uploadDir = path.join(path.resolve(), 'uploads');
if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
}

app.use('/uploads', express.static('uploads'));

app.use('/upload', uploadRoutes);


app.get('/app', (req, res) => {
  res.send('Hello')
})

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
