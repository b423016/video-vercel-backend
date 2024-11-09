import express, { json } from 'express';
import cors from 'cors';
import { PORT, CORS_ORIGIN } from './config/serverConfig';
import uploadRoutes from './routes/uploadRoutes';

const app = express();

app.use(cors({
  origin: CORS_ORIGIN,
  credentials: true
}));
app.use(json());

app.use('/api', uploadRoutes);

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});