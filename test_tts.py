
import asyncio
import edge_tts

async def main():
    text = "Xin chào! Tôi là hệ thống hỗ trợ người khiếm thị. Tôi có thể nhận diện các vật dụng xung quanh bạn."

    print("Dang tao giong noi...")

    communicate = edge_tts.Communicate(
        text,
        voice="vi-VN-HoaiMyNeural"
    )

    await communicate.save("test.mp3")

    print("Da tao xong file test.mp3")

asyncio.run(main())
