import styles from "./page.module.css";
import AudioTool from "./components/audio_tool/AudioTool";

export default function Home() {
  return (
    <div className={styles.page}>
      <main className={styles.main}>
        <AudioTool />
      </main>
      <footer className={styles.footer}>
        <p>Shining Sounds &copy; {new Date().getFullYear()}</p>
      </footer>
    </div>
  );
}
