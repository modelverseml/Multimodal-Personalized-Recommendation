import { price } from '../format'
import { ProductImage, Rating } from './ui'

export default function ProductCard({ product, onOpen, children, rank }) {
  return (
    <article className="product-card">
      <button type="button" className="product-hit" onClick={() => onOpen(product)} aria-label={`Open ${product.title}`}>
        <div className="product-media">
          <ProductImage product={product} />
          {rank != null && <span className="rank-chip">#{rank}</span>}
        </div>
        <div className="product-body">
          <div className="product-meta">
            <span className="badge">{product.category}</span>
            {price(product.price) && <span className="price">{price(product.price)}</span>}
          </div>
          <h3 className="product-title">{product.title}</h3>
          <Rating value={product.average_rating} count={product.rating_number} />
        </div>
      </button>
      {children && <div className="product-extra">{children}</div>}
    </article>
  )
}
