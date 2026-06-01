/**
 * Custom URL handler for webrtcperf → local LiveKit SFU.
 *
 * Generates a LiveKit JWT token for each session and returns the URL
 * pointing to the local demo app.
 *
 */

import { AccessToken } from 'livekit-server-sdk'

const LIVEKIT_URL = process.env.LIVEKIT_URL || 'ws://localhost:7880'
const API_KEY = process.env.LIVEKIT_API_KEY || 'devkey'
const API_SECRET = process.env.LIVEKIT_API_SECRET || 'secret'
const ROOM_NAME = process.env.LIVEKIT_ROOM || 'test-room'
const APP_URL = process.env.APP_URL || 'http://localhost:8080'

export default async function ({ id }) {
  const identity = `webrtcperf-${id}`

  const token = new AccessToken(API_KEY, API_SECRET, {
    identity,
    ttl: '1h',
  })
  token.addGrant({ roomJoin: true, room: ROOM_NAME, canPublish: true, canSubscribe: true })
  const jwt = await token.toJwt()

  return `${APP_URL}?url=${encodeURIComponent(LIVEKIT_URL)}&token=${jwt}`
}
