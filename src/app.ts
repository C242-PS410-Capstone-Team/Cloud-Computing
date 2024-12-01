import express, { Express, Request, Response } from "express";
import dotenv from "dotenv";

import errorHandler from "./middlewares/errorHandler";
import routes from "./routes/routes";

dotenv.config();
const app: Express = express();
const port = process.env.PORT;

app.use(express.json());
app.use("/api", routes);

app.get("/", (req: Request, res: Response) => {
  res.send("Currently API is running");
});

app.use(errorHandler);

// Start the server
app.listen(port, () => {
  console.log(`[server]: Server is running at http://localhost:${port}`);
});
