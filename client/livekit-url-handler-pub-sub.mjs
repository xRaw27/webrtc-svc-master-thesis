/**
 * URL handler for scenarios that split publisher and subscriber roles.
 *
 * Sessions listed in PUBLISHER_SESSIONS (env var, range string) get
 * canPublish: true; the rest get canPublish: false. All sessions can subscribe.
 *
 * PUBLISHER_SESSIONS format: "0", "0,2", "0-1" etc. Defaults to "0".
 */

import { AccessToken } from 'livekit-server-sdk'

const LIVEKIT_URL = process.env.LIVEKIT_URL || 'ws://localhost:7880'
const API_KEY = process.env.LIVEKIT_API_KEY || 'devkey'
const API_SECRET = process.env.LIVEKIT_API_SECRET || 'secret'
const ROOM_NAME = process.env.LIVEKIT_ROOM || 'test-room'
const APP_URL = process.env.APP_URL || 'http://localhost:8080'
const PUBLISHER_SESSIONS = process.env.PUBLISHER_SESSIONS || '0'

function isPublisher(sessionIndex, rangeStr) {
  const parts = String(rangeStr).split(',')
  for (const part of parts) {
    if (part.includes('-')) {
      const [start, end] = part.split('-').map(Number)
      if (sessionIndex >= start && sessionIndex <= end) return true
    } else {
      if (sessionIndex === Number(part)) return true
    }
  }
  return false
}

export default async function ({ id }) {
  const sessionIndex = Number(id)
  const identity = `webrtcperf-${sessionIndex}`
  const canPublish = isPublisher(sessionIndex, PUBLISHER_SESSIONS)

  const token = new AccessToken(API_KEY, API_SECRET, {
    identity,
    ttl: '1h',
  })
  token.addGrant({
    roomJoin: true,
    room: ROOM_NAME,
    canPublish,
    canSubscribe: true,
  })
  const jwt = await token.toJwt()

  return `${APP_URL}?url=${encodeURIComponent(LIVEKIT_URL)}&token=${jwt}`
}
