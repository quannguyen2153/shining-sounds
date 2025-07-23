import { NextRequest } from "next/server";
import ytdl from "@distube/ytdl-core";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url);
  const url = searchParams.get("url");
  if (!url) {
    return new Response("Missing 'url' query parameter", { status: 400 });
  }

  let info;
  try {
    info = await ytdl.getInfo(url);
  } catch (e) {
    return new Response("Failed to get video info", { status: 400 });
  }

  const format = ytdl.chooseFormat(info.formats, { quality: "highestaudio", filter: "audioonly" });
  if (!format || !format.url) {
    return new Response("No audio format found", { status: 404 });
  }

  try {
    const audioStream = ytdl(url, { format });

    const webStream = new ReadableStream({
      start(controller) {
        audioStream.on("data", (chunk) => controller.enqueue(chunk));
        audioStream.on("end", () => controller.close());
        audioStream.on("error", (err) => controller.error(err));
      },
      cancel() {
        audioStream.destroy();
      }
    });

    const contentType = format.mimeType || "audio/mpeg";
    return new Response(webStream, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": `attachment; filename=audio.${contentType.split("/")[1] || "mp3"}`,
        "Cache-Control": "no-store"
      }
    });
  } catch (e) {
    return new Response("Error streaming audio", { status: 500 });
  }
}
