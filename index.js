import express from 'express';
import uploadRoutes from './routes/uploadRoutes.js';
import { config } from 'dotenv';

config();

const app = express();

app.use('/uploads', express.static('uploads'));

app.use('/api/upload', uploadRoutes);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
