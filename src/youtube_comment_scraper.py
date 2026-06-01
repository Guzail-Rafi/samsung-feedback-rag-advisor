import os
import time
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =========================
# 1. LOAD API KEY
# =========================

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found. Please add it inside your .env file.")


# =========================
# 2. BUILD YOUTUBE API CLIENT
# =========================

youtube = build(
    "youtube",
    "v3",
    developerKey=YOUTUBE_API_KEY
)


# =========================
# 3. VIDEO IDS FOR PROJECT
# =========================

VIDEO_IDS = [
    "3i1OB6wKYms",
    "3hPoEmlBQdY",
    "2gYqEi8-am4",
    "nFqEOB8M0eY",
    "Tb0nFak5j-c",
    "_9sBc08FYEU",
    "a4NJNdHqs_I",
    "L84MN-gK3nI",
    "vkBQW8NZk9k",
    "2HhrudHXaDs"
]


# =========================
# 4. GET VIDEO METADATA
# =========================

def get_video_details(video_id):
    """
    This function gets information about the video itself.
    Example: title, channel, views, likes, comments, description, tags.
    """

    try:
        request = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        )

        response = request.execute()

        if not response.get("items"):
            print(f"No video found for ID: {video_id}")
            return None

        item = response["items"][0]

        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})

        return {
            "video_id": video_id,
            "video_title": snippet.get("title", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "video_published_at": snippet.get("publishedAt", ""),
            "video_view_count": statistics.get("viewCount", 0),
            "video_like_count": statistics.get("likeCount", 0),
            "video_comment_count": statistics.get("commentCount", 0),
            "video_description": snippet.get("description", ""),
            "video_tags": ", ".join(snippet.get("tags", []))
        }

    except HttpError as error:
        print(f"Error getting video details for {video_id}")
        print(error)
        return None


# =========================
# 5. GET REPLIES
# =========================

def get_replies(parent_comment_id, video_details):
    """
    This function gets replies under one top-level comment.
    """

    replies = []
    next_page_token = None

    while True:
        try:
            request = youtube.comments().list(
                part="snippet",
                parentId=parent_comment_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )

            response = request.execute()

            for item in response.get("items", []):
                reply = item.get("snippet", {})

                replies.append({
                    "video_id": video_details["video_id"],
                    "video_title": video_details["video_title"],
                    "channel_title": video_details["channel_title"],
                    "video_published_at": video_details["video_published_at"],
                    "video_view_count": video_details["video_view_count"],
                    "video_like_count": video_details["video_like_count"],
                    "video_comment_count": video_details["video_comment_count"],
                    "video_description": video_details["video_description"],
                    "video_tags": video_details["video_tags"],

                    "comment_id": item.get("id", ""),
                    "parent_comment_id": parent_comment_id,
                    "comment_type": "reply",

                    "author": reply.get("authorDisplayName", ""),
                    "comment_text": reply.get("textDisplay", ""),
                    "like_count": reply.get("likeCount", 0),
                    "reply_count": 0,
                    "comment_published_at": reply.get("publishedAt", ""),
                    "comment_updated_at": reply.get("updatedAt", ""),
                    "is_public": True
                })

            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break

            time.sleep(0.3)

        except HttpError as error:
            print(f"Could not collect replies for comment: {parent_comment_id}")
            print(error)
            break

    return replies


# =========================
# 6. GET COMMENTS FOR ONE VIDEO
# =========================

def get_comments_for_video(video_id):
    """
    This function collects top-level comments and replies for one video.
    """

    video_details = get_video_details(video_id)

    if video_details is None:
        return []

    print("\n====================================")
    print(f"Collecting from video: {video_details['video_title']}")
    print(f"Video ID: {video_id}")
    print("====================================")

    all_comments = []
    next_page_token = None

    while True:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
                order="time"
            )

            response = request.execute()

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                top_level_comment = snippet.get("topLevelComment", {})
                top_comment_snippet = top_level_comment.get("snippet", {})

                top_comment_id = top_level_comment.get("id", "")
                total_reply_count = snippet.get("totalReplyCount", 0)

                # Save top-level comment
                all_comments.append({
                    "video_id": video_details["video_id"],
                    "video_title": video_details["video_title"],
                    "channel_title": video_details["channel_title"],
                    "video_published_at": video_details["video_published_at"],
                    "video_view_count": video_details["video_view_count"],
                    "video_like_count": video_details["video_like_count"],
                    "video_comment_count": video_details["video_comment_count"],
                    "video_description": video_details["video_description"],
                    "video_tags": video_details["video_tags"],

                    "comment_id": top_comment_id,
                    "parent_comment_id": "",
                    "comment_type": "top_level",

                    "author": top_comment_snippet.get("authorDisplayName", ""),
                    "comment_text": top_comment_snippet.get("textDisplay", ""),
                    "like_count": top_comment_snippet.get("likeCount", 0),
                    "reply_count": total_reply_count,
                    "comment_published_at": top_comment_snippet.get("publishedAt", ""),
                    "comment_updated_at": top_comment_snippet.get("updatedAt", ""),
                    "is_public": snippet.get("isPublic", True)
                })

                # Collect replies if available
                if total_reply_count > 0:
                    replies = get_replies(top_comment_id, video_details)
                    all_comments.extend(replies)

            next_page_token = response.get("nextPageToken")

            if not next_page_token:
                break

            time.sleep(0.5)

        except HttpError as error:
            print(f"Could not collect comments for video: {video_id}")
            print(error)
            break

    print(f"Collected {len(all_comments)} rows from this video.")
    return all_comments


# =========================
# 7. MAIN FUNCTION
# =========================

def main():
    all_data = []

    for video_id in VIDEO_IDS:
        video_comments = get_comments_for_video(video_id)
        all_data.extend(video_comments)

        # Small delay to avoid sending requests too fast
        time.sleep(1)

    df = pd.DataFrame(all_data)

    if df.empty:
        print("No comments collected.")
        return

    # Remove duplicate comments/replies
    df = df.drop_duplicates(subset=["comment_id"])

    # Remove empty comments
    df = df.dropna(subset=["comment_text"])
    df = df[df["comment_text"].str.strip() != ""]

    # Create output folder
    os.makedirs("data/raw", exist_ok=True)

    # Save final CSV
    output_path = "data/raw/youtube_comments.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n====================================")
    print("DATA COLLECTION COMPLETED")
    print("====================================")
    print(f"Total rows collected: {len(df)}")
    print(f"Saved to: {output_path}")

    print("\nTop-level comments vs replies:")
    print(df["comment_type"].value_counts())

    print("\nRows collected per video:")
    print(df["video_title"].value_counts())

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()